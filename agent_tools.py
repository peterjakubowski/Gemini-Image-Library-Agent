# Google Gemini Image Library Agent
#
# Author: Peter Jakubowski
# Date: 2/1/2025
# Description: Functions, classes, tools, and configurations
#

from google import genai
from google.genai.errors import ClientError
from google.genai import types
from sklearn.neighbors import NearestNeighbors
from pydantic import RootModel
import json
from PIL import Image
from image_utils.image_utils import rescale_width_height
from image_library_db import db, assets, asset_metadata, generative_metadata, embeddings
from schemas import Assets, AssetMetadata, GenerativeMetadata, Embeddings, DescriptionResponseSchema
from pillow_metadata.metadata import Metadata
import numpy as np
import streamlit as st
from io import BytesIO

# =========================
# ===== Gemini Setup ======
# =========================

# Get our api key
# Set up gemini client
try:
    client = genai.Client(api_key=st.secrets['GOOGLE_API_KEY'])
except ClientError as ce:
    raise ce


# =========================
# ======= Functions =======
# =========================


# Open the image from the file path
def open_image(_image_path: str) -> Image.Image:
    _image = Image.open(_image_path)
    # resize the image
    w, h = rescale_width_height(width=_image.size[0], height=_image.size[1], size=1024)

    return _image.resize(size=(w, h))


# Read some metadata from the image file
def read_metadata(_image: Image.Image, _record: Assets) -> AssetMetadata:
    _meta = Metadata(_image)
    _extracted = {'capture_date': _meta.get_capture_date(),
                  'description': _meta.metadata.dc.description,
                  'keywords': ", ".join(_keywords) if (_keywords := _meta.metadata.dc.subject) else _keywords,
                  'creator': _meta.metadata.exif.Artist,
                  'person_in_image': ", ".join(_person_in_image) if (
                      _person_in_image := _meta.metadata.Iptc4xmpExt.PersonInImage) else _person_in_image,
                  'location': _meta.metadata.Iptc4xmpCore.Location,
                  'city': _meta.metadata.photoshop.City,
                  'state': _meta.metadata.photoshop.State}

    return AssetMetadata(**_extracted, id=_record.id)


# Functions to help us insert and update records in the database
def insert_asset(_asset: Assets, _update: bool) -> Assets | None:
    # check if this asset is already in the database
    if not (query := db.q(f"SELECT id FROM assets WHERE file_name == '{_asset.file_name}'")):
        _record = assets.insert(_asset)

        return Assets(**_record)

    elif _update:
        _asset.id = query[0]['id']
        _record = assets.update(_asset)

        return Assets(**_record)

    return None


def insert_metadata(_asset_metadata: AssetMetadata) -> AssetMetadata:
    # if asset metadata is not in the database for this record, then insert it
    if not db.q(f"SELECT id FROM asset_metadata WHERE id == {_asset_metadata.id}"):
        _record = asset_metadata.insert(_asset_metadata)
    # otherwise update the record in the database
    else:
        _record = asset_metadata.update(_asset_metadata)

    return _record


def insert_genai_description(_genai_desc: GenerativeMetadata) -> GenerativeMetadata:
    # if genai description is not in the database for this record, then insert it
    if not db.q(f"SELECT id FROM generative_metadata WHERE id == {_genai_desc.id}"):
        _record = generative_metadata.insert(_genai_desc)
    # otherwise update the record in the database
    else:
        _record = generative_metadata.update(_genai_desc)

    return _record


def insert_embedding(_embedding: Embeddings) -> Embeddings:
    # if embedding is not in the database for this record, then insert it
    if not db.q(f"SELECT id FROM embeddings WHERE id == {_embedding.id}"):
        _record = embeddings.insert(_embedding)
    # otherwise update the record in the database
    else:
        _record = embeddings.update(_embedding)

    return _record


# Wrap our tools in an import workflow
def import_image(_image_path: str, _update=False) -> str:
    """
    Open an image, get information and metadata about the image, and import it into a database.

    :param _image_path:
    :param _update:
    :return:
    """

    # Create an Assets object for our image
    _asset = Assets(image_path=_image_path, file_name=_image_path.split('/')[-1])
    # let's open the image
    _image = open_image(_asset.image_path)
    # check if the image already exists in the database
    if _record := insert_asset(_asset, _update):
        # get the image's embedded metadata
        _img_meta = read_metadata(_image, _record)
        # insert the metadata into the database
        _meta_record = insert_metadata(_img_meta)
        # generate image descriptions
        # time.sleep(1)
        _genai_desc = GeminiDescription(_image,
                                        # RootModel[AssetMetadata](_meta_record).model_dump_json(exclude_none=True),
                                        _meta_record,
                                        _record
                                        ).description
        # insert the generated image descriptions into the database
        _genai_desc_record = insert_genai_description(_genai_desc)
        # generate an embedding
        _embedding = GeminiEmbedding(_description=_genai_desc.description, _record=_record).embedding
        # insert embedding data into the database
        _embedding_record = insert_embedding(_embedding)

        return f"Success! Asset with file name {_asset.file_name} has been updated in the database."

    elif not _update:
        return (f"Asset with file name {_asset.file_name} already exists in database. "
                "If you would like to update the record for this asset, "
                "submit your request again with the update flag set to True.")


def update_assets(_asset: Assets) -> Assets:
    """
    Update a record in the assets table within the database.

    :param _asset:
    :return:
    """

    _record = assets.update(_asset)

    return _record


def update_asset_metadata(_asset_metadata: AssetMetadata) -> AssetMetadata:
    """
    Update a record in the asset_metadata table within the database.

    :param _asset_metadata:
    :return:
    """

    _record = asset_metadata.update(_asset_metadata)

    return _record


def update_genai_description(_genai_desc: GenerativeMetadata) -> GenerativeMetadata:
    """
    Update a record in the generative_metadata table within the database.

    :param _genai_desc:
    :return:
    """

    _record = generative_metadata.update(_genai_desc)

    return _record


def update_embedding(_embedding: Embeddings) -> Embeddings:
    """
    Update a record in the embeddings table within the database.

    :param _embedding:
    :return:
    """

    _record = embeddings.update(_embedding)

    return _record


# Sql search
def search_image_library_sql(_sql_query: str) -> dict:
    """
    Search the image library database using sql queries.

    Example: 'SELECT * FROM assets WHERE id == 1'

    Args:
        _sql_query (string): A sql command to execute on the image library database.

    """

    _query_results = db.q(_sql_query)

    return _query_results


def search_image_library_semantic(_query_text: str) -> list[Assets]:
    """
    Search the image library using vector search.
    The query text should describe an image that we're looking for.
    Accepts query as text, generates an embedding representation of the query,
    and returns results from a nearest neighbors search.
    Revise the query text using other descriptions from the generative metadata table as inspiration.


    Args:
        _query_text (string): Query text string

    """

    print(f"semantic query: {_query_text}")

    # Generate an embedding of our query text
    _response = client.models.embed_content(model="models/text-embedding-004",
                                            contents=_query_text)
    # Format our embedding as a numpy array
    _embedding = np.array(_response.embeddings[0].values).astype(np.float32)
    # Query our nearest neighbors model
    _results = neighbors.search(search_vector=_embedding)[1].tolist()[0]

    _idx = [neighbors.index_map[_res] for _res in _results]

    return [Assets(**search_image_library_sql(f"SELECT * FROM assets WHERE id == {_id}")[0]) for
            _id in _idx]


def python_code_execution(_prompt: str):
    """
    Use a Gemini model to run python and execute code.

    :param _prompt: Text prompt
    :return:
    """

    _response = GeminiCodeExecution(_prompt=_prompt)

    _results = []

    for part in _response.result.candidates[0].content.parts:
        if part.text is not None:
            _results.append(part.text)
        if part.executable_code is not None:
            code_html = f'<pre style="background-color: green;">{part.executable_code.code}</pre>'  # Change code color
            _results.append(code_html)
        if part.code_execution_result is not None:
            _results.append(part.code_execution_result.output)
        if part.inline_data is not None:
            img = Image.open(BytesIO(part.inline_data.data))
            st.chat_message("assistant", avatar=st.session_state.avatars['assistant']).write(img)
            st.session_state.messages.append({"role": "assistant", "content": img})

    return "\n".join(_results)


def retrieve_and_display_image(_id: int) -> str:
    """
    Search the database for the path of the image with the given id.
    Display the image in the chat.

    :param _id: Integer asset id
    :return: String success message
    """

    _result = db.q(f"SELECT * FROM assets WHERE id == '{_id}'")

    if _result:
        img = Image.open(_result[0]['image_path'])
        st.chat_message("assistant", avatar=st.session_state.avatars['assistant']).write(img)
        st.session_state.messages.append({"role": "assistant", "content": img})
        return f"Image id {_id} displayed successfully."

    return f"Path for image id: {_id} could not be retrieved."


def date_string_from_timestamp(_id: int) -> str:
    """
    Search the database for the capture date of the image with the given id.
    Format the date as a string '%A, %B %d, %Y'.

    :param _id:Integer asset id
    :return: String date
    """

    _result = db.q(f"SELECT * FROM asset_metadata WHERE id == '{_id}'")

    if _result and (capture_date := _result[0]['capture_date']):
        return capture_date.strftime('%A, %B %d, %Y')

    return "Could not convert the timestamp to a date string."


# =========================
# ===== Function Tools ====
# =========================

import_image_func = types.FunctionDeclaration(
    name='import_image',
    description='Open an image, get information and metadata about the image, and import it into a database.',
    parameters=types.Schema(
        type=types.Type('OBJECT'),
        properties={
            "_image_path": types.Schema(type=types.Type('STRING')),
            "_update": types.Schema(type=types.Type('BOOLEAN'))
        })
)

update_assets_func = types.FunctionDeclaration(
    name='update_assets',
    description='Update a record in the assets table within the database.',
    parameters=types.Schema(
        type=types.Type('OBJECT'),
        properties={
            "id": types.Schema(type=types.Type('STRING')),
            "image_path": types.Schema(type=types.Type('STRING')),
            "file_name": types.Schema(type=types.Type('STRING')),
            "file_type": types.Schema(type=types.Type('STRING')),
        })
)

update_asset_metadata_func = types.FunctionDeclaration(
    name='update_asset_metadata',
    description='Update a record in the asset_metadata table within the database.',
    parameters=types.Schema(
        type=types.Type('OBJECT'),
        properties={
            "id": types.Schema(type=types.Type('STRING')),
            "capture_date": types.Schema(type=types.Type('STRING')),
            "description": types.Schema(type=types.Type('STRING')),
            "keywords": types.Schema(type=types.Type('STRING')),
            "creator": types.Schema(type=types.Type('STRING')),
            "person_in_image": types.Schema(type=types.Type('STRING')),
            "location": types.Schema(type=types.Type('STRING')),
            "city": types.Schema(type=types.Type('STRING')),
            "state": types.Schema(type=types.Type('STRING'))
        })
)

update_genai_description_func = types.FunctionDeclaration(
    name='update_genai_description',
    description='Update a record in the generative_metadata table within the database.',
    parameters=types.Schema(
        type=types.Type('OBJECT'),
        properties={
            "id": types.Schema(type=types.Type('STRING')),
            "description": types.Schema(type=types.Type('STRING')),
            "keywords": types.Schema(type=types.Type('STRING')),
            "style": types.Schema(type=types.Type('STRING')),
            "mood": types.Schema(type=types.Type('STRING'))
        })
)

search_image_library_sql_func = types.FunctionDeclaration(
    name='search_image_library_sql',
    description=("Search the image library sqlite database using sql queries. "
                 f"Database schema: {db.schema}"
                 ),
    parameters=types.Schema(
        type=types.Type('OBJECT'),
        properties={
            "_sql_query": types.Schema(type=types.Type('STRING'))
        })
)

search_image_library_semantic_func = types.FunctionDeclaration(
    name='search_image_library_semantic',
    description=("Search the image library using vector search."
                 "The query text should describe an image that we're looking for. "
                 "Accepts query as text, generates an embedding representation of the query, "
                 "and returns results from a nearest neighbors search. "
                 "Revise the query text using other descriptions from the generative metadata table as inspiration."
                 ),
    parameters=types.Schema(
        type=types.Type('OBJECT'),
        properties={
            "_query_text": types.Schema(type=types.Type('STRING'))
        })
)

python_code_execution_func = types.FunctionDeclaration(
    name='python_code_execution',
    description=("Use a Gemini model to run python and execute code."
                 ),
    parameters=types.Schema(
        type=types.Type('OBJECT'),
        properties={
            "_prompt": types.Schema(type=types.Type('STRING'))
        })
)

retrieve_and_display_image_func = types.FunctionDeclaration(
    name='retrieve_and_display_image',
    description=("Search the database for the path of the image with the given id."
                 "Display the image in the chat."
                 ),
    parameters=types.Schema(
        type=types.Type('OBJECT'),
        properties={
            "_id": types.Schema(type=types.Type('INTEGER'))
        })
)

date_string_from_timestamp_func = types.FunctionDeclaration(
    name='date_string_from_timestamp',
    description=("Search the database for the capture date of the image with the given id."
                 "Format the date as a string '%A, %B %d, %Y'."
                 ),
    parameters=types.Schema(
        type=types.Type('OBJECT'),
        properties={
            "_id": types.Schema(type=types.Type('INTEGER'))
        })
)

tools = types.Tool(function_declarations=[import_image_func,
                                          update_assets_func,
                                          update_asset_metadata_func,
                                          update_genai_description_func,
                                          search_image_library_sql_func,
                                          search_image_library_semantic_func,
                                          python_code_execution_func,
                                          retrieve_and_display_image_func,
                                          date_string_from_timestamp_func
                                          ]
                   )

# =========================
# ===== Gemini Config =====
# =========================

# Load taxonomy lists
with open('data/style_taxonomy.json') as style_f:
    STYLE_TAX = json.load(style_f)

with open('data/mood_taxonomy.json') as mood_f:
    MOOD_TAX = json.load(mood_f)

# Load system instructions
with open('data/system_instructions_descriptions_model.txt', 'r') as si_desc:
    SYSTEM_INSTRUCTIONS_FOR_DESCRIPTIONS = si_desc.read()

# Load chat system instructions
with open('data/system_instructions_chat_model.txt', 'r') as si_chat:
    CHAT_SYSTEM_INSTRUCTIONS = si_chat.read()

# Configure Gemini model for description generation
GEMINI_DESCRIPTION_CONFIG = types.GenerateContentConfig(safety_settings=None,
                                                        system_instruction=(f"{SYSTEM_INSTRUCTIONS_FOR_DESCRIPTIONS}"
                                                                            f"Style taxonomy: {STYLE_TAX['taxonomy']}"
                                                                            f"Mood taxonomy: {MOOD_TAX['taxonomy']}"),
                                                        max_output_tokens=2048,
                                                        temperature=0.3,
                                                        top_p=0.6,
                                                        top_k=32,
                                                        presence_penalty=0.3,
                                                        frequency_penalty=0.3,
                                                        response_mime_type='application/json',
                                                        response_schema=DescriptionResponseSchema
                                                        )

# Configure Gemini model for embeddings
EMBED_MODEL_CONFIG = types.EmbedContentConfig(output_dimensionality=768,
                                              task_type="SEMANTIC_SIMILARITY")

# Configure Gemini model for code execution
CODE_EXECUTION_CONFIG = types.GenerateContentConfig(system_instruction=("Use python and matplotlib to run code "
                                                                        "and visualize data. "
                                                                        "I give you data, you return a graph."
                                                                        ),
                                                    tools=[types.Tool(code_execution=types.ToolCodeExecution())])

# Configure our Gemini chat model
CHAT_MODEL_CONFIG = types.GenerateContentConfig(safety_settings=None,
                                                tools=[tools],
                                                tool_config=types.ToolConfig(
                                                    function_calling_config=types.FunctionCallingConfig(
                                                        mode=types.FunctionCallingConfigMode("AUTO"))),
                                                system_instruction=CHAT_SYSTEM_INSTRUCTIONS,
                                                max_output_tokens=2048,
                                                temperature=0.0,
                                                top_p=0.6,
                                                top_k=32,
                                                presence_penalty=0.3,
                                                frequency_penalty=0.3,
                                                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                                    disable=True,
                                                    maximum_remote_calls=None
                                                )
                                                )


# =========================
# ===== Gemini Models =====
# =========================


# Configure Gemini model for inference
class GeminiDescription:
    def __init__(self, _image: Image.Image, _image_metadata: AssetMetadata, _record: Assets, ):
        self.image = _image
        self.image_metadata = RootModel[AssetMetadata](_image_metadata).model_dump_json(exclude_none=True)
        self.record = _record
        self.description = self.generate_description()

    def generate_description(self) -> GenerativeMetadata:
        _result = client.models.generate_content(model='gemini-2.0-flash-001',
                                                 config=GEMINI_DESCRIPTION_CONFIG,
                                                 contents=[self.image_metadata, self.image])
        _result_dict = json.loads(_result.text)

        return GenerativeMetadata(id=self.record.id,
                                  description=_result_dict['description'],
                                  keywords=", ".join(_result_dict['keywords']),
                                  style=_result_dict['style'],
                                  mood=_result_dict['mood'],
                                  image_type=_result_dict['image_type'],
                                  subject=_result_dict['subject'],
                                  context=_result_dict['context'],
                                  details=_result_dict['details'],
                                  lighting=_result_dict['lighting'],
                                  framing=_result_dict['framing'],
                                  lens_and_camera=_result_dict['lens_and_camera'],
                                  )


# Configure Gemini model for embedding
class GeminiEmbedding:
    def __init__(self, _description: str, _record: Assets):
        self.description = _description
        self.record = _record
        self.embedding = self.generate_embedding()

    def generate_embedding(self) -> Embeddings:
        _result = client.models.embed_content(model="models/text-embedding-004",
                                              contents=self.description,
                                              config=EMBED_MODEL_CONFIG)
        _embedding = np.array(_result.embeddings[0].values).astype(np.float32).tobytes()

        return Embeddings(id=self.record.id,
                          genai_description_vector=_embedding)


# Configure Gemini for code execution
class GeminiCodeExecution:
    def __init__(self, _prompt: str):
        self.prompt = _prompt
        self.result = None
        self.execute()

    def execute(self):
        self.result = client.models.generate_content(model="models/gemini-2.0-flash-001",
                                                     config=CODE_EXECUTION_CONFIG,
                                                     contents=self.prompt)


# Configure Gemini chat
class GeminiChat:
    def __init__(self):
        self.chat = client.chats.create(model='gemini-2.0-flash-001',
                                        config=CHAT_MODEL_CONFIG)


# =========================
# ==== Neighbors Model ====
# =========================

class Neighbors:

    def __init__(self):
        self.model = None
        self.index_map = {}
        self.train_model()

    def train_model(self):
        vectors_query = db.q("SELECT * FROM embeddings")
        if vectors_query:
            self.index_map = {}
            vectors = []
            for i, v in enumerate(vectors_query):
                self.index_map[i] = v['id']
                vectors.append(np.frombuffer(v['genai_description_vector'], dtype=np.float32))

            _neighbors = NearestNeighbors(n_neighbors=5, radius=1.0)

            self.model = _neighbors.fit(np.array(vectors))

    def search(self, search_vector):
        search_result = self.model.radius_neighbors(search_vector.reshape(1, -1), sort_results=True)
        return search_result


neighbors = Neighbors()


def process_response(_response):
    if function_calls := _response.function_calls:
        results = []
        content = ""
        for call in function_calls:
            try:
                if call.name == 'import_image':
                    content = import_image(**call.args)
                elif call.name == 'search_image_library_sql':
                    content = search_image_library_sql(**call.args)
                elif call.name == 'search_image_library_semantic':
                    content = search_image_library_semantic(_query_text=call.args['_query_text'])
                elif call.name == 'update_assets':
                    content = update_assets(Assets(**call.args))
                elif call.name == 'update_asset_metadata':
                    content = update_asset_metadata(AssetMetadata(**call.args))
                elif call.name == 'update_genai_description':
                    content = update_genai_description(GenerativeMetadata(**call.args))
                elif call.name == 'python_code_execution':
                    content = python_code_execution(**call.args)
                elif call.name == 'retrieve_and_display_image':
                    content = retrieve_and_display_image(**call.args)

            except Exception as ext:
                st.warning(ext.__str__())
                content = ext.__str__()
            finally:
                results.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={'content': content}
                    )
                )
        # time.sleep(1)
        return process_response(st.session_state.chat.send_message(results))

    return [_response.text]
