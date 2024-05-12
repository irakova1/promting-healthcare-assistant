import json
import os.path
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from tqdm import tqdm
from chat import ChatI
import gc

import evaluate as evaluate
import pandas as pd
import matplotlib.pyplot as plt

# import the rouge scorer metric
rouge = evaluate.load('rouge')


def evaluate_model(model_name: str,
                   chat: Union[ChatI, Callable[[], ChatI]],
                   input_texts: List[str],
                   target_outputs: List[str],
                   input_contexts: List[str] = [],
                   results_folder: str = './results/',
                   verbose: bool = False,
                   timeout: int = 0, **params_to_save) -> Path:

    if callable(chat):
        gc.collect()
        time.sleep(2)
        print("Instationate chat")
        chat = chat()

    run_name = "test_eval_" + model_name + "_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    chat.chain_with_history = chat.chain_with_history.with_config(
        {"run_name": run_name}
    )

    model_output = []
    times = []

    if len(input_contexts) < 1:
        input_contexts = [''] * len(input_texts)

    for i, (input_text, target_text, context) in tqdm(enumerate(zip(input_texts, target_outputs, input_contexts)), total=len(input_texts)):
        # timeout for API
        time.sleep(timeout)
        session_id = f'test_{i}'
        chat.clear_session_memory(session_id)
        if isinstance(context, str) and len(context):
            # chat.add_messages(session_id, [('human', context), ('ai', 'Understood. How can I help?')])
            chat.add_messages(session_id, [('human', context + " Use this as context for next questions.")])
        else:
            chat.add_messages(session_id, [('human', "Forget all of provided context. Write answers for unknown humman.")])

        start_time = time.time()
        answer = chat.get_answer(session_id, input_text)
        exe_time = time.time() - start_time
        times.append({'time': exe_time})

        chat.clear_session_memory(session_id)

        model_output.append(answer)

    df = pd.DataFrame(times)

    results = rouge.compute(predictions=model_output,
                            references=target_outputs,
                            use_aggregator=True, )
    if verbose:
        print(f'Model {model_name}: \n{results}\n')

    # Save results
    hyperparams = {"model": model_name, "prompt_config": str(chat.prompt_config), 'time_avr': df.mean()['time']}
    path = evaluate.save(results_folder,
                         # experiment="",
                         **results,
                         **hyperparams,
                         **params_to_save)

    predictions_with_references = pd.DataFrame({"predictions": model_output, "references": target_outputs})
    predictions_with_references.to_csv(str(path).replace(".json", ".csv"))

    return path


def extract_data(path: str) -> Dict:
    with open(path, 'r') as f:
        report = json.load(f)

    return report


def compare(data: List[Dict[str, Any]], by: List[str], output_folder: Optional[str] = None) -> Path:
    from evaluate.visualization import radar_plot

    if output_folder is None:
        output_folder = './results/comparison/'

    df = pd.DataFrame(data)
    model_names = df['model']
    results = df[by].to_dict('records')
    time_columns = [c for c in by if 'time' in c.lower()]

    plot = radar_plot(data=results, model_names=model_names, invert_range=time_columns)
    # plot.show()

    name = '_'.join(map(lambda x: x[:3], model_names)) + f'_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
    # create path if not exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    path_img = os.path.join(output_folder, name + '.png')
    path_df = os.path.join(output_folder, name + '.csv')

    # save results in scv and plot
    plt.savefig(path_img, bbox_inches='tight')
    df.to_csv(path_df)

    return Path(path_img)


def compare_rouge(eval_paths: List[str | Path], by=None, folder: str = None) -> Path:

    if by is None:
        by = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'time_avr']

    data = list(map(extract_data, eval_paths))

    path = compare(data, by=by, output_folder=folder)

    return path
