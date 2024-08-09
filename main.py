# /new-user: user-id, json file for each step  
# step 1: retrieve 30 documents & fileter by LLM evaluation, and provide top 10 documents of relevant documents, provide statistics
# step 2: Generating Q for 25, each generate 1 question, demonstrate top 10 instructions generated, provide statistics
# step 3: sending the question pairs to agent for answering the Q queries, demonstrate the top 10 answers generated. 
# parse the dataframe to json and send to front end. 

from flask import Flask, request, jsonify
from utils import *
import pickle
import os 
from datetime import datetime

app = Flask(__name__)

# global variable that stores all current user information
user_info = dict()


@app.route('/test', methods=['GET'])
def test_route():
    return jsonify('ok'), 200

@app.route('/new-user', methods=['POST'])
def new_user_route():
    """
    returns the information dict of new user
    """
    try:
        current_timestamp = str(datetime.now())
        user_id = current_timestamp
        user_info[user_id] = {
            "user_id": user_id,
            "query": None,
            "is_finished_submit_query": False,
            "is_finished_retrieve_and_filter": False,
            "is_finished_gen_q": False,
            "is_finished_gen_a": False
        }
        return jsonify({'user_info': user_info[user_id]}), 200
    except Exception as e:
        return jsonify("something went wrong"), 400

@app.route('/delete-user', methods=['POST'])
def delete_user_route():
    """
    returns the information dict of all rest of users
    """
    try:
        user_id = request.json.get("user_id")
        if user_id not in user_info:
            return jsonify({'error message': 'user not found'}), 400
        del user_info[user_id]
        delete_files_with_prefix("user_data", user_id)
        return jsonify({'user_info': user_info}), 200
    except Exception as e:
        print(e)
        return jsonify("something went wrong"), 400

@app.route('/get-all-user-info', methods=['GET'])
def get_all_user_info_route():
    """
    returns the information dict of all users
    """
    try:
        return jsonify({'user_info': user_info}), 200
    except Exception as e:
        print(e)
        return jsonify("something went wrong"), 400

@app.route('/get-user-info', methods=['GET'])
def get_user_info_route():
    """
    get information of a single user by user_id
    """
    try:
        user_id = request.json.get("user_id")
        if user_id not in user_info:
            return jsonify({'error message': 'user not found'}), 400
        return jsonify({'user_info': user_info[user_id]}), 200
    except Exception as e:
        print(e)
        return jsonify("something went wrong"), 400

@app.route('/submit-query', methods=['POST'])
def submit_query_route():
    """
    update user info with new query, returns user info of user_id
    """
    try:
        user_id = request.json.get("user_id")
        if user_id not in user_info:
            return jsonify({'error message': 'user not found'}), 400
        query = request.json.get("query")
        if query == None or len(query) == 0:
            return jsonify({'error message': 'Query should not be empty'}), 400
        user_info[user_id]["query"] = query 
        user_info[user_id]["is_finished_submit_query"] = True
        return jsonify({'user_info': user_info[user_id]}), 200
    except Exception as e:
        print(e)
        return jsonify("something went wrong"), 400

@app.route('/retrieve-and-filter', methods=['POST'])
def retrieve_and_filter_route():
    """
    update user info, store the contexts a user generate with the query, returns user info of user_id
    """
    try:
        user_id = request.json.get("user_id")

        # previous steps must be completed
        if user_id not in user_info:
            return jsonify({'error message': 'user not found'}), 400
        if not user_info[user_id]["is_finished_submit_query"]:
            return jsonify({'error message': 'query not submitted'}), 400
        
        remaining_texts = get_texts_by_query(user_info[user_id]['query'], 25)
        stat_dict = {
            "len_contexts": len(remaining_texts)
        }
        with open(f"user_data/{user_id}_contexts.pkl", "wb") as file:
            pickle.dump(remaining_texts, file)
        user_info[user_id]["is_finished_retrieve_and_filter"] = True
        return jsonify({'user_info': user_info[user_id], 'contexts': remaining_texts, 'stats': stat_dict}), 200
    except Exception as e:
        print(e)
        return jsonify("something went wrong"), 400

@app.route('/gen-q', methods=['POST'])
def gen_q_route():
    """
    update user info, generate instructions, returns user info of user_id, and top 10 result
    """
    try:
        user_id = request.json.get("user_id")
        # previous steps must be completed
        if user_id not in user_info:
            return jsonify({'error message': 'user not found'}), 400
        if not user_info[user_id]["is_finished_submit_query"]:
            return jsonify({'error message': 'query not submitted'}), 400
        if not user_info[user_id]["is_finished_retrieve_and_filter"]:
            return jsonify({'error message': 'contexts not found'}), 400
        with open(f"user_data/{user_id}_contexts.pkl", "rb") as file:
            remaining_texts = pickle.load(file)
        all_q = gen_m_q_for_n_context(remaining_texts, 1, n1=20, n2=20, max_attempt=5)
        stat_dict = {
            "len_input_context": len(remaining_texts),
            "len_generated": len(all_q)
        }
        all_q.to_csv(f"user_data/{user_id}_all_q.csv", index=False)
        user_info[user_id]["is_finished_gen_q"] = True
        if len(all_q) > 10:
            top_10 = all_q[:10]
        else:
            top_10 = all_q
        return jsonify({'user_info': user_info[user_id], 'top_10': top_10.to_dict(), 'stats' : stat_dict}), 200
    except Exception as e:
        print(e)
        return jsonify("something went wrong"), 400


@app.route('/gen-a', methods=['POST'])
def gen_a_route():
    """
    update user info, generate answer, returns user info of user_id, and top 10 result
    """
    try:
        user_id = request.json.get("user_id")
        # previous steps must be completed
        if user_id not in user_info:
            return jsonify({'error message': 'user not found'}), 400
        if not user_info[user_id]["is_finished_submit_query"]:
            return jsonify({'error message': 'query not submitted'}), 400
        if not user_info[user_id]["is_finished_retrieve_and_filter"]:
            return jsonify({'error message': 'contexts not found'}), 400
        if not user_info[user_id]["is_finished_gen_q"]:
            return jsonify({'error message': 'instructions not found'}), 400
        all_q = pd.read_csv(f"user_data/{user_id}_all_q.csv")
        completed_df = search_for_all_queries(all_q, user_info[user_id]["query"])
        completed_df.to_csv(f"user_data/{user_id}_completed_df.csv", index=False)
        stat_dict = {
            "len_input_instructions": len(all_q),
            "len_response_generated": len(completed_df)
        }
        if len(completed_df) > 10:
            top_10 = completed_df[:10]
        else:
            top_10 = completed_df
        user_info[user_id]["is_finished_gen_a"] = True
        return jsonify({'user_info': user_info[user_id], 'top_10': top_10.to_dict(), 'stats': stat_dict}), 200
    except Exception as e:
        print(e)
        return jsonify("something went wrong"), 400

@app.route('/get-complete-result', methods=['GET'])
def gen_complete_result_route():
    """
    update user info, store the contexts a user generate with the query, returns user info of user_id
    """
    try:
        user_id = request.json.get("user_id")
        # previous steps must be completed
        if user_id not in user_info:
            return jsonify({'error message': 'user not found'}), 400
        if not user_info[user_id]["is_finished_submit_query"]:
            return jsonify({'error message': 'query not submitted'}), 400
        if not user_info[user_id]["is_finished_retrieve_and_filter"]:
            return jsonify({'error message': 'contexts not found'}), 400
        if not user_info[user_id]["is_finished_gen_q"]:
            return jsonify({'error message': 'instructions not found'}), 400
        if not user_info[user_id]["is_finished_gen_a"]:
            return jsonify({'error message': 'completed result not found'}), 400
        completed_df = pd.read_csv(f"user_data/{user_id}_completed_df.csv")
        stat_dict = {
            "len_results": len(completed_df),
        }
        return jsonify({'user_info': user_info[user_id], 'completion': completed_df.to_dict(), 'stats': stat_dict}), 200
    except Exception as e:
        print(e)
        return jsonify("something went wrong"), 400
    
@app.route('/reset-user-after-completion', methods=['POST'])
def reset_user_after_completion_route():
    try:
        user_id = request.json.get("user_id")
        # previous steps must be completed
        if user_id not in user_info:
            return jsonify({'error message': 'user not found'}), 400
        if not user_info[user_id]["is_finished_gen_a"]:
            return jsonify({'error message': 'cannot reset incomplete user'}), 400
        delete_files_with_prefix("user_data", user_id)
        user_info[user_id] = {
            "user_id": user_id,
            "query": None,
            "is_finished_submit_query": False,
            "is_finished_retrieve_and_filter": False,
            "is_finished_gen_q": False,
            "is_finished_gen_a": False
        }
        return jsonify({'user_info': user_info[user_id]}), 200
    except Exception as e:
        print(e)
        return jsonify("something went wrong"), 400

if __name__ == '__main__':
    os.makedirs("user_data", exist_ok=True)
    delete_all_contents("user_data")
    app.run(host='0.0.0.0', port=8000, debug=True)