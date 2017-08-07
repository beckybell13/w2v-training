from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

TOPIC_DIR = 'topic_with_sentence_per_line.txt'
OUTPUT_FILE_NAME = 'w2v-topic-model'

topic_file = open(TOPIC_DIR)
sentences = LineSentence(topic_file)

# NOTE: cython must be installed for parallelization
model = Word2Vec(sentences, size=100, min_count=1, workers=4)
model.save(OUTPUT_FILE_NAME)
