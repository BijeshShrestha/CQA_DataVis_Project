import base64
import streamlit as st
from streamlit import pyplot as plt
from streamlit_pdf_viewer import pdf_viewer
from CQP_MVP import *
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from PIL import Image
# Initialize OpenAI model
llm = OpenAI(model="gpt-3.5-turbo")

# Initialize AgentRunner
agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_index.as_retriever(similarity_top_k=5),
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=True,
)
agent = AgentRunner(agent_worker)

def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="100" height="50%" type="application/pdf"
                        style="height:50vh; width:50%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Streamlit user interface setup
st.set_page_config(
    page_title="CQA Tool",
    page_icon=":toolbox:",
    layout="wide"
)

st.markdown("""
<style>
iframe {
    width: 110% !important;
    height: 8000px !important;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.title('📊 Chart Question Answering')

# add PDFs by uploading in the sidebar
with st.sidebar:
    st.markdown("### Upload PDF :page_facing_up:")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    # add status indicator
    if uploaded_file:
        st.markdown("### Status :hourglass_flowing_sand:")
        st.success("Ready to answer questions")
        # st.balloons()

# Here, switch the order of col1 and col2 to reverse the panes
col1, col2 = st.columns(2)  # Adjusted the order here
# Display PDF
if uploaded_file:
    with col2.expander("📈 PDF Preview"):
        display_pdf(uploaded_file)
col2.markdown("###     CQA Tool Framework")
col2.image(Image.open("./img/draft_pipeline.png"))
col2.markdown("###     CQA Tool Instructions")
col2.markdown("1️⃣ Upload a PDF file")
col2.markdown("2️⃣ Ask a question about the content of the PDF file")    
col2.markdown("3️⃣ The tool will generate a response to your question")

with col2.expander(":warning:  Finding and Limitations "):
    st.markdown("1️⃣ The tool is designed to answer questions about the content of the PDF file")
    st.markdown("2️⃣ The tool may not be able to answer all questions")
    st.markdown("3️⃣ The tool may not be able to answer questions that require external knowledge")
    st.markdown("4️⃣ The tool may not be able to answer questions that require complex reasoning")


# # Initialize variables to store PDF data
# if 'text_data' not in st.session_state or 'chart_data' not in st.session_state:
#     st.session_state['text_data'], st.session_state['chart_data'] = [], []

# Ensure conversation and response history are part of session state
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'response_history' not in st.session_state:
    st.session_state['response_history'] = []

# Display the conversation history dynamically
def display_conversation_history():
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []
    if 'response_history' not in st.session_state:
        st.session_state['response_history'] = []

    for i in range(len(st.session_state['conversation'])):
        col1.write(f"User: {st.session_state['conversation'][i]}")
        if i < len(st.session_state['response_history']):
            col1.write(f"{st.session_state['response_history'][i]}")

        # Additionally check for image responses and display them
        if i < len(st.session_state['response_history']) and isinstance(st.session_state['response_history'][i], str) and st.session_state['response_history'][i].endswith('.png'):
            st.image(st.session_state['response_history'][i], caption="Generated Chart")


# Display the conversation history and any images first
display_conversation_history()

def send_message():
    user_message = st.session_state['current_message']
    if user_message:
        # Append message to conversation
        st.session_state['conversation'].append(user_message)

        # Generate a response using the agent, which might include generating an image
        latest_image_path = process_inquiry_and_show_latest_image(user_message)
        if latest_image_path:
            # If an image is generated, use the image path as the response to display
            st.session_state['response_history'].append(latest_image_path)
            st.image(latest_image_path, caption="Latest Generated Chart")

        # Otherwise, generate a text response using the agent
        else:
            response = agent.chat(user_message)
            st.session_state['response_history'].append(response)

        # Clear the current message to reset the text input box
        st.session_state['current_message'] = ""

        # # Optionally rerun to update the UI
        # st.rerun()

if 'current_message' not in st.session_state:
    st.session_state['current_message'] = ""

# # Display the conversation history and any images first
# display_conversation_history()

# Input and send buttons are defined after displaying the images
message_input = col1.text_input("Enter your message here", key='current_message')
send_button = col1.button("Send", on_click=send_message)

if send_button:
    st.rerun()
