import logging.config
import os
import re
import sys
import codecs
import gensim
import mediacloud

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join('data', 'w2v-topic-models')

# set up logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.info("-------------------------------------------------------------------")

try:
	API_KEY = os.environ['API_KEY']
except KeyError:
	logging.error('You need to define the API_KEY environment variable.')
	sys.exit(0)

mc = mediacloud.api.AdminMediaCloud(API_KEY)

TOPIC_ID = sys.argv[1]

STORIES_PER_FETCH_DEFAULT = 100
MODEL_SIZE_DEFAULT = 100
MIN_COUNT_DEFAULT = 1
NUM_WORKERS_DEFAULT = 4

def download_topic_sentences(output_file_path):
	logger.info('Grabbing topic sentences...')

	query = '{~ topic:'+TOPIC_ID+'}'
	logger.info('Query: ' + query)

	stories_per_fetch = os.getenv('STORIES_PER_FETCH', STORIES_PER_FETCH_DEFAULT)
	logger.info('Stories per fetch: {}'.format(stories_per_fetch))

	story_count = mc.storyCount(query)['count']
	total_pages = (story_count / stories_per_fetch) + 1
	logger.info('Total stories to fetch: {}'.format(story_count))
	logger.info('Total pages: {}'.format(total_pages))

	logger.info('Fetching pages...')
	with codecs.open(output_file_path, 'w', 'utf-8') as f:
		last_processed_stories_id = 0
		more_stories = True
		page = 0
		while more_stories:
			logger.info('Page {} of {}'.format(page, total_pages))
			stories = mc.storyList(query, last_processed_stories_id=last_processed_stories_id, rows=stories_per_fetch,
			                       sentences=True)
			more_stories = len(stories) > 0
			if len(stories) > 0:
				last_processed_stories_id = stories[-1]['processed_stories_id']

			for story in stories:
				for sentence in story['story_sentences']:
					sent = re.sub(r'[^\w\s-]', '', sentence['sentence']).lower()
					f.write(re.sub(r'[\n\r]', ' ', sent))
					f.write('\n')
			page += 1

		logger.info('Done!')


def train_topic(sentences, output_file_name):
	# NOTE: cython must be installed for parallelization
	logger.info('Training w2v model for topic...')

	model_size = os.getenv('SIZE', MODEL_SIZE_DEFAULT)
	min_count = os.getenv('MIN_COUNT', MIN_COUNT_DEFAULT)
	num_workers = os.getenv('WORKERS', NUM_WORKERS_DEFAULT)

	model = gensim.models.Word2Vec(sentences, size=model_size, min_count=min_count, workers=num_workers)
	model.save(output_file_name)

	logger.info('Done!')


# Create word2vec topic model:
if __name__ == '__main__':
	# grab sentences (setup export file so we can write as we fetch)
	sentences_file_path = os.path.join(BASE_DIR, OUTPUT_DIR, TOPIC_ID+'-sentences-.txt')
	download_topic_sentences(sentences_file_path)

	# train w2v topic model
	model_file_path = os.path.join(BASE_DIR, OUTPUT_DIR, 'w2v-topic-model-' + TOPIC_ID)
	with codecs.open(sentences_file_path, 'r', 'utf-8') as f:
		sentences = gensim.models.word2vec.LineSentence(f)
		train_topic(sentences, model_file_path)

	# remove sentences file
	os.remove(sentences_file_path)
