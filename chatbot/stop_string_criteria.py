# code copied from https://github.com/huggingface/transformers/pull/28932
# transformers/src/transformers/generation/stopping_criteria.py

from collections import OrderedDict
from typing import Optional, List, Tuple, Union, Dict

import numpy as np
from torch.nn import functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import StoppingCriteria
import torch

STOP_STRING_EMBEDDING_CACHE = OrderedDict()


class StopStringCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever specific string sequences are generated. It preprocesses
    the strings together with the tokenizer vocab to find positions where tokens can validly complete the stop strings.

    Generation is stopped as soon as a token is generated that completes any of the stop strings.
    We want to catch any instance in which the stop string would be present in the decoded output, which means
    we must also catch cases with "overhangs" off one or both ends. To make this more concrete, for the stop string
    "stop", any of the following token sequences would trigger the match:

    - ["st", "op"]
    - ["stop"]
    - ["st", "opera"]
    - ["sto", "opper"]
    - ["las", "topper"]

    Note that a match will only be triggered if the stop string is at the end of the generated sequence. In other
    words, these sequences will not trigger a match:

    - ["stop", "at"]
    - ["st", "op", "at"]
    - ["st", "opera", "tion"]

    The reason these are not a match is that the stop string does not overlap with the final token. If you can remove
    one or more tokens from the end of the sequence without destroying the stop string, then this criterion will not
    match that stop string. This is by design; because this check is run after each token is generated, we can't miss a
    valid stop string if one is generated, but we don't want to halt generation just because the stop string exists
    somewhere in the past input_ids.

    How is the match actually performed, though? We do it in quite a confusing way, because we want the entire match
    process to be compilable with Torch or XLA, which means we cannot use standard string methods. However, it is possible,
    with some work, to do string matching with pure tensor operations. We'll begin by describing the algorithm we use
    with standard string operations, and then at the end we'll explain how this is converted to pure tensor operations.

    The key to the algorithm is an observation: Because the stop string must overlap with the end of the token sequence, we can start at
    the end of the sequence and work backwards. Specifically, we check that there is an overlap between the start of
    the final token and the end of the stop_string, or to put it another way, stop_string[-i:] == token[:i] for
    some i > 0. If you look at the positive examples above, you'll see the last token in all of them fulfills this
    property:

    - ["st", "op"] (overlap is "op", overlap length == 2)
    - ["stop"]  (overlap is "stop", overlap length == 4)
    - ["st", "opera"]  (overlap is "op", overlap length == 2)
    - ["sto", "pper"]  (overlap is "p", overlap length == 1)
    - ["las", "topper"]  (overlap is "top", overlap length == 3)

    It's impossible to construct a matching sequence that does not have this property (feel free to verify this
    yourself). However, although this overlap between the start of the final token and the end of the stop string is
    necessary for a match, it is not sufficient. We also need to check that the rest of the token sequence is
    consistent with the stop string.

    How do we do that? Let's say the stop string is N characters long, and the initial overlap covers the final
    M characters. Then, we have N - M characters left to match. If the next token is less than N - M tokens long, then
    the entire token must match: token == stop_string[-M - len(token): -M]. If the next token is longer than N - M
    tokens, then we consider only the final N - M characters of the token. This allows for the token to have an overhang
    off the start of the stop string.

    Again, let's make this concrete with a worked example. We'll use the stop string "stop" and the token sequence
    ["las", "topper"]. The length of the stop string is 4. The final token is "topper", and its overlap with the stop
    string is "top", which has length 3. We continue to the next token, "las", and we have 4 - 3 = 1 character left to
    match. This is less than the length of "las", so we only need a partial match for this token to complete the string.
    We check that "las"[-1:] == stop[:1], which is true. We have now matched 4 characters, which is the length of
    the stop string, and we are done.

    At this point, hopefully you agree that we have an algorithm that detects the presence of a stop string, but you
    may not see how we can convert this to tensor operations, particularly since we want to avoid data-dependent
    conditional branching in the compiled code, and ideally vectorize everything so it can be efficiently computed on
    GPU. The key is to realize that although we don't have access to string operations inside the generation loop,
    we can use them in a precomputation stage!

    For every token in the tokenizer vocabulary, we precompute the values
    we need for the above algorithm: The length of that token's overlap with the end of the stop string, the
    position(s) in the stop string where that token matches perfectly, and the length of the token. We then pack
    these values into a single vector per token, and stack those vectors into an embedding tensor which we can
    gather from at runtime to get the values we need.

    This is the approach we take in this class. The precomputation is done in the `_stop_string_create_embedding_vec`
    function. Then, at runtime in the `__call__()` method, we implement the algorithm above in purely vectorized
    fashion, starting from an input_ids vector containing the token IDs in the sequence:

    - Gather from the embedding vector using input_ids as indices, and split the packed vectors into end overlap lengths,
      valid token positions, and token lengths.
    - Make a vector of the length of every token in the sequence, except for the final token, where we use the
      end-overlap length instead.
    - Compute the cumulative sum of the sequence, starting from the end. This represents the number of characters in the stop string that
      we would match after each token, assuming that token is a valid fit for the sequence at that point.
    - To determine if the tokens are valid at each position, we check that the cumulative length so far matches
      one of the values in their valid positions vector. Where it does not, we mask that token and all tokens
      preceding it.
    - We then check the highest unmasked value in the cumulative sum. This represents the length of the total string
      match before we reached a token that did not match the stop string. If it is equal to or greater than the length
      of the stop string, the stop string is matched.

    This is almost the complete algorithm, and the remaining details just handle edge cases: For example, what do
    we do if a token can have multiple possible overlap lengths with the stop string? For example, consider the
    stop string "banana", and the token sequences ["ba", "nana"] and ["bana", "nana"]. Both of these sequences
    contain the stop string and should trigger a match. However, the overlap of the final token is different! In
    the first case, the overlap is "nana". In the second case, the overlap is "na". When we start from the end
    of the sequence and work backwards, we cannot know in advance which overlap length, if any, will lead to a valid
    match, and therefore we must test all possible overlap lengths.

    Therefore, for the stop string "banana" and the token "nana", we store two valid end overlap lengths: 2 and 4.
    We then perform the above algorithm, starting from each value, and test whether each results in a match.
    Thanks to vectorization, we can run these tests in parallel (in fact, we can run the test for every possible
    overlap length and all stop strings in parallel).

    The second detail is how we handle cases when the token sequence has an overhang off the start of the stop string,
    as in the case of ["las", "top"], since we do not store "start overlaps" in the same way we do for end overlaps.
    Instead, we simply store (in the valid_positions vector) that the token "las" is valid before "top", in the same
    way that the token "s" is. Therefore, the total length it computes in the case of ["las", "top"] is 6 rather than 4,
    because it doesn't truncate the match to the length of the stop string. However, since the algorithm concludes by
    checking that the maximum match length is equal to or greater than the length of the stop string, this does not
    affect the correctness of its final answer; both ["las", "top"] with a total length of 6, and ["s", "top"] with a
    total length of 4, will be correctly identified as matches, because both are >= 4.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            The model's associated tokenizer (necessary to extract vocab and tokenize the termination sequences)
        stop_strings (`Union[str, List[str]]`):
            A list of strings that should end generation. If a string is passed, it will be treated like a
            list with a single element.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    >>> model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    >>> inputs = tokenizer("The biggest states in the USA by land area:", return_tensors="pt")

    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    The biggest states in the USA by land area:
    - Alaska
    - Texas
    - California

    >>> # Passing one or more stop strings will halt generation after those strings are emitted
    >>> # Note that generating with stop strings requires you to pass the tokenizer too
    >>> gen_out = model.generate(**inputs, stop_strings=["Texas"], tokenizer=tokenizer)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    The biggest states in the USA by land area:
    - Alaska
    - Texas
    ```
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, stop_strings: Union[str, List[str]]
    ):
        if isinstance(stop_strings, str):
            stop_strings = [stop_strings]
        self.stop_strings: Tuple[str, ...] = tuple(stop_strings)
        vocab = tokenizer.get_vocab()
        token_list, token_indices = tuple(vocab.keys()), tuple(vocab.values())
        self.embedding_vec, self.max_valid_positions, self.max_valid_end_lens = (
            self.clean_and_embed_tokens_with_cache(
                token_list, token_indices, self.stop_strings, tokenizer
            )
        )

        self.maximum_token_len = max(
            [len(stop_string) for stop_string in self.stop_strings]
        )
        self.num_stop_strings = len(self.stop_strings)
        self.target_lens = torch.tensor(
            [len(stop_string) for stop_string in stop_strings], dtype=torch.int32
        )

    def clean_and_embed_tokens_with_cache(
        self, token_list, token_indices, stop_strings, tokenizer
    ):
        # We don't use the tokenizer in the cache key, because I don't trust it to have well-behaved equality
        if (token_list, token_indices, stop_strings) in STOP_STRING_EMBEDDING_CACHE:
            embedding_vec, max_valid_positions, max_valid_end_lens = (
                STOP_STRING_EMBEDDING_CACHE[
                    (token_list, token_indices, self.stop_strings)
                ]
            )
            STOP_STRING_EMBEDDING_CACHE.move_to_end(
                (token_list, token_indices, stop_strings)
            )
        else:
            clean_token_list, clean_token_indices = self.clean_tokenizer_vocab(
                tokenizer
            )
            embedding_vec, max_valid_positions, max_valid_end_lens = (
                self._stop_string_create_embedding_vec(
                    clean_token_list, clean_token_indices, stop_strings
                )
            )
            STOP_STRING_EMBEDDING_CACHE[(token_list, token_indices, stop_strings)] = (
                embedding_vec,
                max_valid_positions,
                max_valid_end_lens,
            )
            if len(STOP_STRING_EMBEDDING_CACHE) > 8:
                STOP_STRING_EMBEDDING_CACHE.popitem(
                    last=False
                )  # Pop from the start, the least recently used item
        return embedding_vec, max_valid_positions, max_valid_end_lens

    @staticmethod
    def clean_tokenizer_vocab(tokenizer, static_prefix="abcdef"):
        """
        This method turns a tokenizer vocab into a "clean" vocab where each token represents the actual string
        it will yield, without any special prefixes like "##" or "Ġ". This is trickier than it looks - the method
        tokenizer.convert_tokens_to_string() does not always return the correct string because of issues with prefix
        space addition/removal. To work around this, we add a static prefix to the start of the token, then remove
        it (and any prefix that may have been introduced with it) after calling convert_tokens_to_string().
        """
        vocab = tokenizer.get_vocab()
        clean_token_list = []
        clean_token_indices = []
        sentence_base = tokenizer(static_prefix, add_special_tokens=False)["input_ids"]
        tokens_base = [tokenizer._convert_id_to_token(tok) for tok in sentence_base]
        for token, token_idx in vocab.items():
            token_string = tokenizer.convert_tokens_to_string(tokens_base + [token])
            token_string = token_string[
                token_string.index(static_prefix) + len(static_prefix) :
            ]
            clean_token_list.append(token_string)
            clean_token_indices.append(token_idx)
        return tuple(clean_token_list), tuple(clean_token_indices)

    @staticmethod
    def _stop_string_get_matching_positions(
        token_list, token_indices, stop_strings
    ) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]]]:
        """This function preprocesses stop strings and the tokenizer vocabulary to determine where tokens can
        validly appear in the stop strings. For each token, it computes a list of positions in the stop string where the
        token appears, as well as a list of the possible "end overlaps" for that token - that is, the number of characters
        from the end of the stop string that overlap with the start of the token, which can have more than one value.

        The reason for computing these may seem a bit cryptic - please see the docstring for StopStringCriteria for a full
        explanation of what these values are for!"""

        token_valid_positions = {}
        token_end_overlaps = {}
        for stop_string in stop_strings:
            reversed_stop_string = stop_string[::-1]
            token_valid_positions[stop_string] = {}
            token_end_overlaps[stop_string] = {}
            for token, tok_idx in zip(token_list, token_indices):
                reversed_token = token[::-1]
                matching_positions = []
                possible_end_lengths = []
                for i in range(1 - len(token), len(stop_string)):
                    if i < 0:
                        tok = reversed_token[-i:]
                        i = 0
                    else:
                        tok = reversed_token
                    stop = reversed_stop_string[i : i + len(tok)]
                    if tok.startswith(stop):
                        if i == 0:
                            possible_end_lengths.append(min(len(tok), len(stop)))
                        else:
                            matching_positions.append(i)

                if matching_positions:
                    token_valid_positions[stop_string][tok_idx] = matching_positions
                if possible_end_lengths:
                    token_end_overlaps[stop_string][tok_idx] = possible_end_lengths
        return token_valid_positions, token_end_overlaps

    @staticmethod
    def _stop_string_create_embedding_vec(
        token_list, token_indices, stop_strings
    ) -> Dict[str, torch.tensor]:
        """This function precomputes everything needed for the run-time checks in StopStringCriteria, and packs
        them into an embedding tensor that can be accessed with pure tensor operations. For the specifics of the values
        that are precomputed and what they are used for, please refer to the StopStringCriteria docstring!
        """
        token_valid_positions, token_end_overlaps = (
            StopStringCriteria._stop_string_get_matching_positions(
                token_list, token_indices, stop_strings
            )
        )

        max_valid_positions = max(
            len(val)
            for positions in token_valid_positions.values()
            for val in positions.values()
        )
        max_valid_end_lens = max(
            len(val)
            for positions in token_end_overlaps.values()
            for val in positions.values()
        )
        vec_size = len(stop_strings) * (max_valid_positions + max_valid_end_lens) + 1
        gather_vec = np.full((len(token_list), vec_size), dtype=np.int32, fill_value=-1)

        for i, stop_string in enumerate(stop_strings):
            positions = token_valid_positions[stop_string]
            end_lens = token_end_overlaps[stop_string]

            # Since this is lots of very small assignments of lists, we build it with numpy rather
            # than torch for speed + simplicity, then convert to torch at the end
            for token_idx, valid_positions in positions.items():
                gather_vec[
                    token_idx,
                    max_valid_positions * i : max_valid_positions * i
                    + len(valid_positions),
                ] = valid_positions
            for token_idx, possible_end_lens in end_lens.items():
                gather_vec[
                    token_idx,
                    max_valid_positions * len(stop_strings)
                    + max_valid_end_lens * i : max_valid_positions * len(stop_strings)
                    + max_valid_end_lens * i
                    + len(possible_end_lens),
                ] = possible_end_lens
            for token, token_idx in zip(token_list, token_indices):
                gather_vec[token_idx, -1] = len(token)

        gather_vec = torch.tensor(gather_vec, dtype=torch.int32)

        return gather_vec, max_valid_positions, max_valid_end_lens

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.Tensor:
        self.embedding_vec = self.embedding_vec.to(input_ids.device)
        self.target_lens = self.target_lens.to(input_ids.device)
        # The maximum length we need to consider is 1 token per character. Note that input_ids can also be
        # *shorter* than the global max, and the code below should be ready for that
        input_ids = input_ids[:, -self.maximum_token_len :]

        # Flip input_ids because we're only matching strings at the end of the generated sequence
        flipped_ids = torch.flip(input_ids, (1,))

        # Size of the vector of positions a single token can match
        max_valid_positions = self.max_valid_positions

        # The embedding vec contains the valid positions, end_lengths and total lengths for each token
        embedded = F.embedding(flipped_ids, self.embedding_vec)

        # Now we split the embedding vector. valid_positions is the positions in the stop string the token can fit
        valid_positions = embedded[
            :, 1:, : max_valid_positions * self.num_stop_strings
        ].unflatten(-1, (self.num_stop_strings, -1))
        # end_lengths is the number of characters from the string, counting from the end, that the token
        # contains. It can have multiple values if the same token can overlap different end lengths
        end_lengths = embedded[
            :, :1, max_valid_positions * self.num_stop_strings : -1
        ].unflatten(-1, (self.num_stop_strings, -1))
        # Lengths is the total length of each token. Unlike the others, it always has a single value
        lengths = embedded[
            :, 1:, None, -1:
        ]  # Insert a dummy dimension for stop_strings even though lengths are const

        # Concatenate lengths onto each possible end_lengths value
        lengths = lengths.expand((-1, -1, end_lengths.shape[-2], end_lengths.shape[-1]))
        lengths_with_ends = torch.cat([end_lengths, lengths], dim=1)

        # cumsum() to get the number of matched characters in the stop string after each token
        cumsum = lengths_with_ends.cumsum(
            dim=1
        )  # B x maximum_token_len x num_stop_strings x max_valid_end_lens

        # The calculation above assumes that all tokens are in valid positions. Now we mask the ones that are not.
        # First, tokens match the start of the string if they have a positive value in the end_lengths vector
        initial_match = end_lengths > 0

        # Tokens continue the string if the cumsum() so far is one of the valid positions for that token
        # Note that we're actually tracking one cumsum() for for each possible end_length
        later_match = torch.any(
            cumsum[:, :-1, :, None] == valid_positions[:, :, :, :, None], axis=-2
        )

        # The match vector is a boolean vector that indicates which positions have valid tokens
        match = torch.cat([initial_match, later_match], dim=1)

        # Once a single position does not match, all positions following that position are masked
        mask = (~match).cumsum(dim=1, dtype=torch.int32)
        mask = mask == 0

        # The string is matched if we reached a cumsum equal to or greater than the length of the string
        # before hitting the mask
        string_matches = (
            torch.amax(cumsum * mask, dim=(1, -1)) >= self.target_lens[None, :]
        )

        # We return a per-sample vector that is True if any stop string is matched for that sample
        # print(f"\nper-sample vector: {torch.any(string_matches, dim=-1)}")
        return torch.any(string_matches, dim=-1)
