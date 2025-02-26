import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Initialize Streamlit UI
st.title("DeepSeek-R1 Chatbot 💬")
st.write("Powered by LangChain, Ollama, and Streamlit")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store messages in session state

# Display chat history
for message in st.session_state.messages:
    role = "🧑‍💻 You" if message["role"] == "user" else "🤖 AI"
    with st.chat_message(role):
        st.write(message["content"], unsafe_allow_html=True)  # Allow HTML styling

# User input field (chat-style input)
user_input = st.chat_input("Ask me anything...")

# Define a prompt template
template = """Question: {question}

Answer: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)

# Load the Ollama model with DeepSeek-R1
model = OllamaLLM(model="deepseek-r1:1.5b", streaming=True)  # Enable streaming response

# Create a LangChain pipeline (prompt → model)
chain = prompt | model

# Process user input
if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("🧑‍💻 You"):
        st.write(user_input)

    # Display AI thinking message
    with st.chat_message("🤖 AI"):
        # Show the <think> tag with styling
        think_message = '<p style="color:gray; font-size:14px;"><think>Thinking here...</think></p>'
        think_placeholder = st.empty()  # Placeholder for updating the text later
        think_placeholder.markdown(think_message, unsafe_allow_html=True)

        # Stream AI response
        response_placeholder = st.empty()
        full_response = ""  # Store streamed response

        with st.spinner("AI is responding..."):
            for chunk in chain.stream({"question": user_input}):  # Stream the response
                full_response += chunk
                response_placeholder.markdown(full_response)  # Update UI dynamically

        # Remove the thinking message after full response
        think_placeholder.empty()

    # Save AI response to chat history
    st.session_state.messages.append({"role": "ai", "content": full_response})