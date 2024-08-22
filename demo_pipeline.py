from utils import * 
import argparse
from datetime import datetime 

# code to create demo dataset 
# queries I will try: finance, politics, computer engineering, art, education
# for each of them, retrieve top 50 contexts and filter them; then for each attempt to generate 5 questions and answer

if __name__ == "__main__":
    # try for queries: [finance, politics, computer engineering, art, education]
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="user query")
    parser.add_argument("--num_texts", type=int, help="number of contexts to query from database")
    parser.add_argument("--num_instructs", type=int, help="number of instructions generated for each context")
    parser.add_argument("--output_dir", type=str, help="directory to store output", default="demo_result")

    args = parser.parse_args()
    
    # initialize retriever
    print("initializing retriever...")
    embeddings = CustomJinaEmbeddings()
    vectordb = Chroma(persist_directory="fineweb_db_new_50k", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": args.num_texts})

    # retrieve num_texts documents and filter
    print("getting contexts...")
    remaining_texts = get_texts_by_query(args.query, args.num_texts, retriever)
    print(f"number of remaining contexts: {len(remaining_texts)}")

    # generate instructions
    print("generating instructions...")
    os.makedirs(f"{args.output_dir}_data", exist_ok=True)
    if os.path.exists(f"{args.output_dir}_data/{args.query}_all_q.csv"):
        all_q = pd.read_csv(f"{args.output_dir}_data/{args.query}_all_q.csv")
    else:
        all_q = gen_m_q_for_n_context(remaining_texts, args.num_instructs, n1=20, n2=20, max_attempt=5)
        all_q.to_csv(f"{args.output_dir}_data/{args.query}_all_q.csv", index=False)
    print(f"number of instructions generated: {len(all_q)}")

    # generate answer
    print("searching the web for answer...")
    completed_df = search_for_all_queries(all_q, args.query)
    print(f"number of answers that are successfully generated: {len(completed_df)}")

    # save result
    print("saving...")
    os.makedirs(args.output_dir, exist_ok=True)
    current_timestamp = datetime.now()
    completed_df.to_csv(f"{args.output_dir}/{args.query}_{current_timestamp}.csv", index=False)

    


    