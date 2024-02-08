import os
import traceback
import logging
import mimetypes
import nltk
from nltk.tokenize import sent_tokenize
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from protecto_ai import ProtectoVault
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
from langchain.vectorstores import Milvus
from langchain_core.documents import Document

# milvus_instance = Milvus(OpenAIEmbeddings)


load_dotenv()

nltk.download('punkt')


class FileQuestioner:

    def __init__(self):
        self.chunk_size = int(os.getenv("CHUNK_SIZE"))
        self.model_name = os.getenv('MODEL_NAME')
        self.max_tokens = int(os.getenv('MAX_TOKENS'))
        self.sentence_count_limit = int(os.getenv('SENTENCE_COUNT_LIMIT'))
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        protecto_api_key = os.getenv("PROTECTO_API_KEY")
        file = os.getenv("FILE_PATH")

        if not file:
            raise ValueError("File path not specified")
        mime_type, _ = mimetypes.guess_type(file)
        if not mime_type == "application/pdf":
            return

        self.pdf_content = extract_text(file)
        self.protecto_object = ProtectoVault(protecto_api_key)
        self.concat_sentence = ''

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filename="Error.log", filemode='w')
        self.logger = logging.getLogger()

    def sentence_tokenize(self, texts):
        try:
            sentence_count = 0
            sentence_appended = ""
            sentences = sent_tokenize(texts)

            for each_line in sentences:
                sentence_appended += each_line + " "
                sentence_count += 1

                if sentence_count >= self.sentence_count_limit:
                    self.logger.info("Masking Started")
                    self.logger.info(f"{sentence_appended}")
                    masked_sentence = self.mask(sentence_appended)
                    self.logger.info("Masking Completed")
                    self.concat_sentence += masked_sentence
                    sentence_appended = ""
                    sentence_count = 0

            if sentence_appended:
                masked_sentence = self.mask(sentence_appended)
                self.concat_sentence += masked_sentence

            self.logger.info("Sentences are MASKED, appended and sent to Process")
            self.process(self.concat_sentence)

        except Exception as e:
            self.logger.error(f"Error in sentence tokenize: {e}\n\n Trace: {traceback.format_exc()}")
            raise e

    def mask(self, sentence_appended):
        # Update the below line with Async call
        # masking_sentence = self.protecto_object.mask({"mask": [{"value": sentence_appended}]})
        # masked_sentence = masking_sentence["data"][0]["token_value"]
        # return masked_sentence
        return sentence_appended

    def split_text_with_overlap(self, text, chunk_size, overlap_word_size):
        try:
            self.logger.info("Splitting text with overlap started")
            # Custom tokenization to respect special tags
            words = []
            current_word = ""
            inside_tag = False

            for char in text:
                if char == '<':
                    inside_tag = True
                    if current_word:
                        words.append(current_word)
                        current_word = ""
                elif char == '>':
                    inside_tag = False
                    current_word += char
                    words.append(current_word)
                    current_word = ""
                    continue

                current_word += char
                if not inside_tag and char.isspace():
                    words.append(current_word)
                    current_word = ""

            if current_word:
                words.append(current_word)

            # Splitting into chunks with overlap
            chunks = []
            current_chunk = []
            current_length = 0

            for word in words:
                current_length += len(word)
                if current_length >= chunk_size:
                    chunks.append(''.join(current_chunk).strip())
                    # chunks.append(Document(page_content=''.join(current_chunk).strip()))
                    current_chunk = current_chunk[-overlap_word_size:]
                    current_length = len(''.join(current_chunk))
                current_chunk.append(word)

            if current_chunk:
                chunks.append(''.join(current_chunk).strip())
                # chunks.append(Document(page_content=''.join(current_chunk).strip()))

            self.logger.info("Splitting text with overlap completed")
            return chunks

        except Exception as e:
            self.logger.error(f'Error in split text with overlap: {e}\n\n Trace: {traceback.format_exc()}')
            raise e

    def process(self, output_masked_sentences):
        try:
            overlap_word_size = 4
            texts = self.split_text_with_overlap(text=output_masked_sentences, chunk_size=self.chunk_size,
                                                 overlap_word_size=overlap_word_size)
            self.logger.info(
                f"Text split with overlap and completed, with {self.chunk_size} as chunk size and {overlap_word_size} "
                f"as overlap")

            embeddings = OpenAIEmbeddings()
            # Store the data in Milvus database
            print('texts', texts)
            # retriever = Milvus(
            #     texts,
            #     embeddings,
            #     connection_args={"host": "localhost", "port": "19530"},
            #     # index_params="text"
            # )
            # return
            connection_args = {"host": "127.0.0.1", "port": "19530"}
            # texts = ["one", "two"]
            COLLECTION_NAME = 'doc_qa_db4'
            retriever = Milvus.from_texts(
                texts=texts,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args=connection_args,
                # index_params="text"
            )

            # (
            #     embedding_function=embeddings,
            #     connection_args=connection_args,
            #     collection_name=COLLECTION_NAME,
            #     drop_old=True,
            # )

            # retriever = Chroma.from_texts(texts, embeddings,)
            self.logger.info('Embedding Created')
            message_history = ChatMessageHistory()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer',
                                              chat_memory=message_history)
            llm = ChatOpenAI(model_name=self.model_name, max_tokens=self.max_tokens)
            bot = ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever.as_retriever(), return_source_documents=True,
                memory=memory, chain_type="stuff")

            while True:
                try:
                    query = input("Enter the question (type 'exit' or Ctrl+c to end): ")
                    if query.lower() == "exit":
                        print("\nExiting.")
                        break
                    val = bot({"question": query})
                    print(val["answer"], "\n")
                except KeyboardInterrupt:
                    print("\nExiting.")
                    break

        except Exception as e:
            self.logger.error(f"Error in Process :{e}\n\n Trace: {traceback.format_exc()}")
            raise e

    def execute(self):
        try:
            self.sentence_tokenize(self.pdf_content)
        except Exception as e:
            self.logger.error(f"Error in execute: {e}\n\n Trace {traceback.format_exc()}")
            raise RuntimeError(f"Error: {e}")


# class Document:
#     def __init__(self, page_content, pk=None):
#         self.page_content = page_content
#         self.pk = pk  # Assign a unique value to each document

obj = FileQuestioner()
obj.execute()