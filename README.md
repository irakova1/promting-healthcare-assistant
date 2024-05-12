# healthworks-chatbot

## Set up

### Install requirements

### Fill up .env

Fill up `.env.example` and delete rename it to `.env`

## Evaluation

In order to evaluate model you'll need some data to evaluate and model. 

### Custom model

#### Download model

Download model in llamafile format.

##### UNIX
```bash
wget https://huggingface.co/shaikatasif/BioMistral-7B-llamafile/resolve/main/BioMistral-7B.Q4_K_M.llamafile
```

Add permission to run the llamafile.

```bash
chmod +x BioMistral-7B.Q4_K_M.llamafile
```

Run model server. There you will find what port model will be listening. Typically, 
it's http://127.0.0.1:8080 or one up if this port taken (eg. http://127.0.0.1:8081 
or other next available port).   

```bash
./BioMistral-7B.Q4_K_M.llamafile --server --nobrowser --embedding -ngl 999
```

Where `-ngl` - number of offloaded layers to GPU. `BioMistral-7B.Q4_K_M` have 33 layers, 28 of which could be offloaded into 4GB VRAM.

In case of `run-detectors: unable to find an interpreter` use solution from https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#gotchas
```bash
sudo wget -O /usr/bin/ape https://cosmo.zip/pub/cosmos/bin/ape-$(uname -m).elf
sudo chmod +x /usr/bin/ape
sudo sh -c "echo ':APE:M::MZqFpD::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"
sudo sh -c "echo ':APE-jart:M::jartsr::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"
```

##### Windows
Windows have limit for executable size, you cant run llamafile with embeded weights. Download llamafile-0.7.exe from `https://github.com/Mozilla-Ocho/llamafile/releases/download/0.7/llamafile-0.7`

```powershell
.\llamafile-0.7.exe -m .\BioMistral-7B.Q4_K_M.llamafile --server --nobrowser --embedding -ngl 999
```

#### Run evaluation example

Now we can run test of evaluation. The reference how to make evaluation and comparing
results of the models you can find in `chatbot/test_eval.py`.

To run evaluation and comparison of the models (will be used the same model with 
different temperature parameters). Run following command.

```bash
python3 chatbot/test_eval.py --model_type llamafile --model_url http://127.0.0.1:8081  --data_path data/mini_test.csv
```

Where `--model_url` you set after what port your model is running on and 
`--data_path` where your test data is. There is `mini_test.csv` file ander `data` 
folder for quick test. 


### Using API

#### Setting up environment

Make sure you have set up `.env` file and your GOOGLE_APPLICATION_CREDENTIALS are 
correct.

Alternatively you could just set up variable for in your console:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=PATH_TO_JSON_FILE
```

Where `PATH_TO_JSON_FILE` you substitute with absolute path to your Google 
credentials json file. 

#### Run evaluation example

To run evaluation and comparison of the models (will be used the same model with 
different temperature parameters). Run following command.

```bash
python3 test_eval.py --model_type vertexai  --data_path ../data/mini_test.csv
```

Where `--model_url` you set after what port your model is running on and 
`--data_path` where your test data is. There is `mini_test.csv` file ander `data` 
folder for quick test. 


## DeepEval test

All it tests, results, and setup file stored in `tests` folder. 

### Set up .env

For using DeepEval tests you will need to fill up in the `.env` file `DeepEval` part,
and OpenAI part, since default *evaluation* model is GPT4. 

DeepEval can use other models for evaluation, but it will need vast output window 
(the more, the better) because all its standard metrics LLM-based, and it requests GPT 
to form output for all test sample in json for each statement. 

Usage different model is NOT implemented yet. You can add it by adding additional param
to the env. Do NOT forget to trac it in the `hyperparameters()` in the test file. 

You also need to set up other `.env` variables:
- `CHAT_MODEL` - type of the model to use in chat. All it possible values are in 
`chatbot/chat.py` `ModelType`.
- `CHAT_MODEL_NAME` - used for models/API without defined model option in the `.env` 
name (as for llamafiles or huggingface).
- `CHAT_MODEL_URL` - URL to the model API (for llamafiles) or for model repo in 
huggingface.
- `CHAT_DB` - URL to the database that will store message history.
- `API_TIME_OUT` - used in `chatbot/test_eval` and `tests` for setting timeout 
between request to the chat model in case of reaching RPM or TPM limit. For VertexAI
its 12 s, for Anthropic 1 second.


## Running tests

After specifying key to API you are going to use and application variables in the 
`.env` you can run the DeepEval test with next command from this repo root:

Login to save eval results to ConfidentAI.
```bash
deepeval login
```

```bash
deepeval test run tests
```
