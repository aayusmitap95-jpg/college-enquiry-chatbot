
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="College Enquiry Chatbot",
    page_icon="🎓",
    layout="centered"
)

# # Load the sample dataset
# @st.cache_data
# def load_data():
#     return pd.read_csv("college_faq.csv")

# df = load_data()

def load_data():
    data = pd.read_csv("college_faq.csv")
    data.columns = data.columns.str.strip()

    data["question"] = data["question"].fillna("").astype(str).str.strip()
    data["answer"] = data["answer"].fillna("").astype(str).str.strip()

    return data

df = load_data()

# # Prepare the NLP model
# vectorizer = TfidfVectorizer()
# question_vectors = vectorizer.fit_transform(df["question"])
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

question_vectors = vectorizer.fit_transform(df["question"])
# Find the best matching question
def get_response(user_question):
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(
        user_vector, question_vectors
    )[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]
    # st.write("Best matching FAQ:", df.iloc[best_index]["question"])
    # st.write("Similarity score:", round(float(best_score), 3))

    if best_score < 0.20:
        return (
            "Sorry, I couldn't find a suitable answer. "
            "Please contact the college office for help."
        )

    return df.iloc[best_index]["answer"]

# Website design
st.title("🎓 College Enquiry Chatbot")
st.write(
    "Welcome! Ask me about admissions, courses, "
    "fees, scholarships, placements, and facilities."
)
st.caption(
    "Educational demo using sample data. "
    "Answers are examples, not verified college information."
)

# Keep conversation history during the session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 How can I help you with "
                "your college enquiry today?"
            )
        }
    ]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept a new question
if prompt := st.chat_input("Type your college question here..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    answer = get_response(prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)
