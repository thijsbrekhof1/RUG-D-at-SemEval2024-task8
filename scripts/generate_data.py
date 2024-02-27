from vllm import LLM, SamplingParams
import os
import json
import argparse
from datasets import load_dataset


def set_hf_home(directory):
    os.environ['HOME'] = directory


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', "--n_pages", default=1000, type=int,
                        help=f"Number of pages to generate (default 1000)")
    parser.add_argument("-lan", "--language", default='english', type=str,
                        help="Language you want generated (default english)")
    parser.add_argument("-s", "--seed", default=1, type=int,
                        help="Random seed to use for generating the pages (default 1)")

    args = parser.parse_args()

    return args


def get_wikipedia_titles(args):
    if args.language == 'german':
        data = load_dataset("wikipedia", "20220301.de")
    else:
        data = load_dataset("wikipedia", "20220301.en")

    new = data["train"].shuffle(seed=args.seed).select(range(args.n_pages))
    print('created new')

    titles, texts = [], []
    for item in new:
        titles.append(item["title"])
        texts.append(item["text"][:1200])
        print(item["title"])

    return titles, texts


def main():
    args = create_arg_parser()
    wikipedia_pages, wikipedia_texts = get_wikipedia_titles(args)

    # Generate prompts
    if args.language == "english":
        prompts = [
            f"Write a Wikipedia article about {page}. The article should contain at least 250 words. Write the article in English."
            for page in wikipedia_pages]

    elif args.language == "german":
        prompts = [
            f"Schreiben Sie einen Wikipedia-Artikel über {page}. Der Artikel sollte mindestens 250 Wörter umfassen. Schreiben Sie den Artikel auf Deutsch."
            for page in wikipedia_pages]

    elif args.language == "arab":
        prompts = [
            f"aktub maqalatan wikibidya ean {page}. yajib 'an tahtawi almuqalat ealaa 250 kalimatan ealaa al'aqala. kitabat almaqal biallughat alearabiati."
            for page in wikipedia_pages]

    elif args.language == "indonesian":
        prompts = [
            f"Tulis artikel Wikipedia tentang {page}. Artikel harus berisi minimal 250 kata. Tulis artikel dalam bahasa Indonesia."
            for page in wikipedia_pages]

    elif args.language == "russian":
        prompts = [
            f"Napishite stat'yu v Vikipedii o {page}. Stat'ya dolzhna soderzhat' ne meneye 250 slov. Napishite stat'yu na russkom yazyke."
            for page in wikipedia_pages]

    else:
        print("Prompt not found for the language provided.")
        exit()

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=400)

    # Create an LLM.
    llm = LLM(model="NousResearch/Llama-2-7b-hf")

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)

    # Write the generated data to a JSONL file
    with open('extra_' + args.language + '.jsonl', 'w') as f:
        for count, (output, text) in enumerate(zip(outputs, wikipedia_texts)):
            json.dump({"text": output.outputs[0].text, "label": 1, "source": "args.language", "id": count}, f)
            f.write('\n')
            json.dump({"text": text, "label": 0, "source": "args.language", "id": count + len(outputs)}, f)
            f.write('\n')


if __name__ == "__main__":
    main()
