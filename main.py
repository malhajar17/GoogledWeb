# /new-user: user-id, json file for each step  
# step 1: retrieve documents & fileter by LLM evaluation, and provide top 10 documents of relevant documents
# step 2: Generating Q for 20, each generate 1 question, demonstrate top 10 instructions generated
# step 3: sending the question pairs to agent for answering the Q queries, demonstrate the top 10 instruction generated. 
# parse the dataframe to json and send to front end. 

# demo dataset 
# finance, politics, computer engineering, art, education
# for each of them, retrieve top 50 contexts and filter them; then for each generate 5 questions and answer