
from . import *
import argparse

parser = argparse.ArgumentParser(description='Generating word embeddings from a vocab file')
parser.add_argument('--vocab', dest='vocab', type=str, help='path to a vocab file (one word per line)')
parser.add_argument('--big', dest='big', type=bool, help='whether to use the big embedding or not for models that support it')
parser.add_argument('--size', dest='size', type=int, help='size of the embedding or size of the window for models that support it')
parser.add_argument('--model', dest='model', type=str, help='model to use, see models folder for the complete list')
parser.add_argument('--artRetriever', dest='art_retriever', type=str, help='either to use wordnet ("wo") summary or wikipedia ("wi") summary')
parser.add_argument('--out', dest='out', type=str, help='path to the output file')

args = parser.parse_args()

if args.vocab is None:
    raise ValueError("Please provide a vocab file")

with open(args.vocab, "r") as f:
    vocab = list(map(lambda x: x.strip().replace("\n", ""), f.readlines()))

if args.big is None:
    args.big = False

if args.size is None:
    args.size = 300

if args.model is None:
    args.model = "bert"

args.model = args.model.lower()
assert args.model in ["bert", "roberta", "glove", "wikipedia2vec", "docbert", "docberta"]

if args.art_retriever is None:
    args.art_retriever = "wi"

if args.model == "roberta":
    model = ROBERTAModel(vocab, big = args.big, window = args.size)
elif args.model == "bert":
    model = BERTModel(vocab, big = args.big, window = args.size)
elif args.model == "wikipedia2vec":
    model = Wiki2VecModel(vocab, size = args.size)
elif args.model == "docbert":
    model = DocBERTModel(vocab, big = args.big)
elif args.model == "docberta":
    model = DocBERTAModel(vocab, big = args.big)
elif args.model == "glove":
    model = GloVEModel(vocab)

if args.out is None:
    from os import sep
    args.out = f"{args.vocab.split(sep)[-1]}{args.model}_{args.size}{'B' if args.big else ''}_{args.art_retriever}"

if args.art_retriever == "wi":
    art_retriever = WikipediaArticleRetriever(args.out + ".art", vocab)
elif args.art_retriever == "wo":
    art_retriever = WordNetArticleRetriever(args.out + ".art", vocab)

if args.model in ["bert", "roberta", "docbert", "docberta"]:
    if art_retriever():
        art_retriever.save()

model.reset_embeddings()

if not model.check_embeddings_exist(args.out + ".csv", art_retriever):
    model.convert(art_retriever)
    model.export(args.out + ".csv")


print("Converted classes: ", len(model.get_class_list()))
