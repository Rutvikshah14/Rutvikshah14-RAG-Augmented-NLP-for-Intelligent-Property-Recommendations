import re
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
print(os.getenv('CUDA_PATH'))
import pickle
from llama_cpp import Llama,LlamaGrammar
from torch import cuda
import hashlib
import json

# print(cuda.is_available())
# print(cuda.current_device())
# print(cuda.device_count())
# print(cuda.get_device_name(0))
# print(cuda.memory_allocated())
# print(cuda.memory_reserved())
# exit()

embedding_model = SentenceTransformer('all-MiniLM-L12-v2', device='cuda')

class AirbnbRAGRecommender:
    def __init__(self, df, model_path, corpus_embeddings, generator_seed=0):

        
        # Load the dataset
        self.df = df
        self.corpus_embeddings = corpus_embeddings

        #prepare llm after embdeddings, to not reserve GPU
        self.llm = Llama(
            model_path = model_path,
            n_ctx = 4000,
            n_batch = 512,
            verbose = True,
            n_gpu_layers = -1,
            max_tokens = -1,
            seed =  generator_seed,
        )
        
  
    def retrieve_similar_listings(self, query, top_k=5):
        """Retrieve top similar listings based on cosine similarity"""
        # Embed the query
        query_embedding = embedding_model.encode([query])

        # Calculate cosine similarities
        embedding_model.similarity_fn_name ='cosine'
        similarities = embedding_model.similarity(query_embedding, self.corpus_embeddings)
        # Get indices of top-k similar listings
        top_indices = np.argsort(similarities[0])[-top_k:]
        
        # Return top listings
        return self.df.iloc[top_indices], similarities[0][top_indices]
    
    def generate_recommendation(self, query):
        """Generate a personalized recommendation using local LLM"""
        #similar listings
        top_listings, similarities = self.retrieve_similar_listings(query)
        
        #context for LLM
        context = "\n\n".join([
            f"Listing {i+1}: {row.listing_text}"
            for i, row in enumerate(top_listings.itertuples(name='Listing'))
        ])

        iterator = '_'.join(query.split(' '))

        open (f"context_{iterator}.txt", 'w',encoding='utf-8') .write(context)
        

        

        prompt = f"""
                Select one of the options below that matches requirement:'{query}' , only describe why and also denote the price to the destination country currency.
                Give your recommendation in json format.
                Options:
                {context}
                

                assistant :"""
        
        import httpx
        grammar_text = httpx.get("https://raw.githubusercontent.com/ggerganov/llama.cpp/master/grammars/json_arr.gbnf").text
        grammar = LlamaGrammar.from_string(grammar_text) #getting json output

        # Generate recommendation using local LLM
        response = self.llm(
            prompt,
            stop=["</s>"],
            echo=False,
            max_tokens=500,
            repeat_penalty=1.5,
            grammar=grammar,
            temperature=0.65,
            top_p=0.5,
        )
        return {
            'top_listings': top_listings,
            'similarities': similarities,
            'recommendation': response['choices'][0]['text'].strip()
        }


def prepare_corpus(embedding_dir='embeddings', df=None):
    """Prepare embeddings, using cached version if available"""
    
    os.makedirs(embedding_dir, exist_ok=True)
    # Generatehash
    dataset_hash = generate_dataset_hash(df)
    
    #pickle files
    embedding_path = os.path.join(embedding_dir, f'{dataset_hash}_embeddings.pkl')
    
    # trying to load existing embeddings
    if os.path.exists(embedding_path):
        try:
            with open(embedding_path, 'rb') as f:
                corpus_embeddings = pickle.load(f)
            print("Loaded existing embeddings successfully.")
            df['listing_text'] = df.apply(create_listing_description, axis=1)
            return corpus_embeddings
        except (pickle.UnpicklingError) as e:
            print(f"Error loading cached embeddings: {e}")
            print("Will regenerate embeddings.")
    
    df['listing_text'] = df.apply(create_listing_description, axis=1)


    # Generate embeddings
    print("Generating embeddings...")
    corpus_embeddings = embedding_model.encode(
        df['listing_text'].tolist(), 
        show_progress_bar=True
    )
    
    # Save embeddings with pickle
    print("Saving embeddings...")
    try:
        with open(embedding_path, 'wb') as f:
            pickle.dump(corpus_embeddings, f)
        print("Embeddings saved successfully.")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

    return corpus_embeddings
def generate_dataset_hash(df):
    """Generate a unique hash for the current dataset"""
    #combination of dataset shape, column names
    dataset_info = (
        str(df.shape) + 
        '_'.join(df.columns)
    )
    return hashlib.md5(dataset_info.encode()).hexdigest()

def create_listing_description(row):
    """Create a comprehensive text description for a listing"""
    description = (
        f"A {row.get('room_type')} {row.get('name')} in {row.get('neighbourhood')},{row.get('city')} "
        f"with {row.get('bedrooms')} beds. "
        f"Amenities include: {row.get('amenities')}. "
        f"Price per night: ${row.get('price')} Local {row.get('city')} currency. "
        f"Rating: {row.get('review_scores_rating')}"
    )
    return description
  
df = pd.read_csv('Listings.csv',encoding_errors="ignore")
print(df.shape)
useful_columns = ['room_type', 'bedrooms','property_type','name','amenities','price','review_scores_rating','neighbourhood','city']
df = df[useful_columns]
df = df.dropna()
df = df.sample(frac=0.7,random_state=42) #randomly sample 70% of remaining data
#save df to csv
df.to_csv('sampled_listings.csv',index=False)
corpus_embeddings = prepare_corpus(df=df)

# model_path = './models/mistral-7b-v0.1.Q6_K.gguf'
model_path = './models/mistral-7b-instruct-v0.2.Q5_K_M.gguf'
model_name = model_path.split('/')[-1]
print(model_name)
generator_seed = 0
# Create recommender (will use cached embeddings if available)
recommender = AirbnbRAGRecommender(df, model_path,corpus_embeddings)

# Example queries
queries = [
    # "Romantic getaway in Paris",
    # "Family-friendly apartment near attractions",
    "family holiday with pet friendly stay",
    "hot tub sauna",
    "Luxury stay with ocean view",
    "cheap in the mountains",
]

os.makedirs('results', exist_ok=True)
        
for query in queries:
    with open(f"results/{query}_recommend.json", 'w', encoding='utf-8') as f:
        recommendation = recommender.generate_recommendation(query)
        json.dump(json.loads(recommendation['recommendation']), indent=4, fp=f)