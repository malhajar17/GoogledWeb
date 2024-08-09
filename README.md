# FineWebSFT

### What is it
- This is our project for NexaAI Hackathon.
- By using GenQA technique in this [paper](https://arxiv.org/pdf/2406.10323), and LLM agent powered by groq and Langchain, we enable generation of up-to-date instruction data for finetuning, base on user input query
- Base dataset of generation is [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Since this is only a demo, we create vector database using JinaEmbedding with only the top 10k rows from fineweb-edu 2024-10 snapshot.

### To use the command line demo
- create your keys.json file following this fomat
    ```json
    {
        "GROQ_API_KEY": "",
        "JINA_API_KEY": "",
        "GOOGLE_CSE_ID": "",
        "GOOGLE_API_KEY": "",
        "HUGGINGFACE_TOKEN": ""
    }
    ```
- create an environment with python version 3.11.3
- ```pip install -r requirements.txt```
- ```python demo_pipeline.py --query <your_query> --num_texts <number_of_texts_you_want_to_retrieve_from_database> --num_instructs <number_of_instructions_you_want_to_generate_for_each_num_texts> --output_dir <directory_to_store_final_result>```
- Note that due to API rate limit, sometimes generation may take longer than expected or fail.
