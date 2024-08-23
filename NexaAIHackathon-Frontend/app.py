import streamlit as st
import pandas as pd
from server_utils import register_new_user, delete_user, submit_query, retrieve_and_filter, gen_q, gen_a, get_complete_result
from utils import convert_df_to_csv, convert_df_to_json, convert_df_to_excel, load_css

# Set page configuration
st.set_page_config(page_title="Query Processing Demo", page_icon=":mag:", layout="centered")

# Apply custom CSS for background and text color
st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFA;
    }
    .stButton>button {
        background-color: #44A1A0;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1A7A7B;
    }
    .stProgress > div > div > div > div {
        background-color: #1A7A7B;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the main and secondary color scheme
MAIN_COLOR = "#1A7A7B"  # Lighter Teal
SECONDARY_COLOR = "#44A1A0"  # Aqua
BACKGROUND_COLOR = "#FFFFFA"  # Off-White

# Title and introduction
st.markdown(f"<h1 style='text-align: center;'>Query Processing Pipeline</h1>", unsafe_allow_html=True)

# Initialize session state variables
if "user_id" not in st.session_state:
    user_response = register_new_user()
    if isinstance(user_response, dict) and "error" in user_response:
        st.error(user_response["error"])
    elif user_response is not None:
        st.session_state["user_id"] = user_response["user_id"]
    else:
        st.error("Unexpected error occurred during user registration.")

if "filter_response" not in st.session_state:
    st.session_state["filter_response"] = None

if "gen_q_response" not in st.session_state:
    st.session_state["gen_q_response"] = None

if "gen_a_response" not in st.session_state:
    st.session_state["gen_a_response"] = None

if "complete_results" not in st.session_state:
    st.session_state["complete_results"] = None

st.markdown(f"""
<div style='text-align: center;'>
    <h3>Welcome to the Query Processing Pipeline!</h3>
    <p>Our advanced system will guide you through the following steps to process your query:</p>
    <ol style='text-align: left; display: inline-block;'>
        <li><strong>Query Input</strong>: Enter your query to get started.</li>
        <li><strong>Search FineWeb</strong>: We search the 50k FineWeb dataset to find relevant contexts.</li>
        <li><strong>Generate Questions</strong>: Extract subjects and generate questions from the contexts using state-of-the-art GENQA approach.</li>
        <li><strong>Search Google</strong>: AI agents answer the complex questions generated.</li>
        <li><strong>View and Download</strong>: Display the final results and download them in multiple formats.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

user_query = st.text_input("Enter your query:", placeholder="e.g., Tell me about knowledge related to medicine")

if st.button("Submit"):
    if user_query:
        with st.spinner("Processing your query..."):
            user_id = st.session_state["user_id"]

            # Initialize a progress bar
            progress = st.progress(0)

            # Step 1: Submit Query
            st.markdown(f"<h4>Step 1: Submitting Your Query</h4>", unsafe_allow_html=True)
            submit_response = submit_query(user_id, user_query)
            if "error" in submit_response:
                st.error(submit_response["error"])
            else:
                st.session_state['submit_response'] = submit_response
                st.success("Your query has been successfully submitted!")
                progress.progress(20)

                # Step 2: Retrieve and Filter Contexts
                st.markdown(f"<h4>Step 2: Searching FineWeb</h4>", unsafe_allow_html=True)
                st.info("Searching the 50k FineWeb dataset for relevant contexts...")
                filter_response = retrieve_and_filter(user_id)

                if "error" in filter_response:
                    st.error(filter_response["error"])
                else:
                    st.session_state['filter_response'] = filter_response
                    num_contexts = filter_response["stats"].get('len_contexts', 'N/A')
                    st.write(f"🔍 Contexts found: {num_contexts}")
                    with st.expander("View contexts"):
                        st.write(st.session_state['filter_response'].get('contexts', "No contexts available"))
                    progress.progress(40)

                    # Step 3: Generate Questions
                    st.markdown(f"<h4>Step 3: Generating Questions</h4>", unsafe_allow_html=True)
                    st.info("Generating questions from the contexts using state-of-the-art GENQA approach...")
                    gen_q_response = gen_q(user_id)
                    if "error" in gen_q_response:
                        st.error(gen_q_response["error"])
                    else:
                        st.session_state['gen_q_response'] = gen_q_response
                        num_questions = gen_q_response['stats'].get('len_generated', 'N/A')
                        st.write(f"📝 Questions generated: {num_questions}")
                        with st.expander("View questions"):
                            st.write(st.session_state['gen_q_response']["top_10"].get('instruction', "No questions available"))
                        progress.progress(60)

                        # Step 4: Generate Answers
                        st.markdown(f"<h4>Step 4: Using AI to Answer Questions</h4>", unsafe_allow_html=True)
                        st.info("Using AI agents to answer the previously generated complex queries...")
                        gen_a_response = gen_a(user_id)
                        if "error" in gen_a_response:
                            st.error(gen_a_response["error"])
                        else:
                            st.session_state['gen_a_response'] = gen_a_response
                            num_answers = gen_a_response['stats'].get('len_response_generated', 'N/A')
                            st.write(f"🤖 Answers generated: {num_answers}")
                            with st.expander("View answers"):
                                st.write(st.session_state['gen_a_response']["top_10"].get('response', "No answers available"))
                            progress.progress(80)

                            # Step 5: Retrieve Complete Results
                            st.markdown(f"<h4>Step 5: Retrieving Final Results</h4>", unsafe_allow_html=True)
                            st.info("Finalizing and compiling the results...")
                            complete_results = get_complete_result(user_id)
                            if "error" in complete_results:
                                st.error(complete_results["error"])
                            else:
                                st.session_state['complete_results'] = complete_results
                                st.write("Final Results:")
                                df = pd.DataFrame(complete_results['completion'])
                                st.dataframe(df)

                                # Provide download options with clear labels
                                st.markdown(f"<h5>Download Your Results:</h5>", unsafe_allow_html=True)
                                csv_data = convert_df_to_csv(df)
                                json_data = convert_df_to_json(df)
                                excel_data = convert_df_to_excel(df)

                                st.download_button(
                                    label="Download data as CSV",
                                    data=csv_data,
                                    file_name='query_results.csv',
                                    mime='text/csv',
                                )

                                st.download_button(
                                    label="Download data as JSON",
                                    data=json_data,
                                    file_name='query_results.json',
                                    mime='application/json',
                                )

                                st.download_button(
                                    label="Download data as Excel",
                                    data=excel_data,
                                    file_name='query_results.xlsx',
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                )

                                st.success("Processing completed successfully!")
                                progress.progress(100)
                                # Add footer
                                st.markdown(
                                    """
                                    <footer>
                                        Developed by Mohamad Alhajar and Barbara Su for Team Googlers in the NexaAI Hackathon
                                    </footer>
                                    """,
                                    unsafe_allow_html=True
                                )
    else:
        st.error("Please enter a valid query.")

# Logout button with confirmation and success message
if st.button("Logout"):
    if "user_id" in st.session_state:
        delete_user(st.session_state["user_id"])
        del st.session_state["user_id"]
        st.success("You have been logged out successfully.")
