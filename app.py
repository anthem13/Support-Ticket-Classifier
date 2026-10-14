"""streamlit run app.py (after python train.py)"""
import streamlit as st

from tickets.model import TicketClassifier

st.set_page_config(page_title="Ticket Triage", page_icon="🎫")
st.title("🎫 Support ticket triage")


@st.cache_resource
def load():
    return TicketClassifier.load("models/ticket_clf.joblib")


clf = load()
text = st.text_area("Paste a ticket", "hi, I got charged twice and need this fixed ASAP!!")
if st.button("Classify"):
    r = clf.predict([text])[0]
    c1, c2 = st.columns(2)
    c1.metric("Category", r["category"], f"{r['category_confidence']:.0%} confident")
    c2.metric("Urgency", r["urgency"], f"{r['urgency_confidence']:.0%} confident")
    if r["needs_human"]:
        st.warning("Low confidence → route to a human agent")
