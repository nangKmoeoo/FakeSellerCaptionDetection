import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Fake Seller Caption Detection",
    page_icon="📦"
)


# DARK / LIGHT MODE SWITCH
theme = st.sidebar.toggle("🌙 Dark Mode", value=True)

if theme:
    dark_css = """
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }

    textarea {
        background-color: #262730 !important;
        color: white !important;
    }

    button {
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 6px;
    }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)

else:
    light_css = """
    <style>
    .stApp {
        background-color: white;
        color: black;
    }

    textarea {
        background-color: white !important;
        color: black !important;
    }

    button {
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 6px;
    }
    </style>
    """
    st.markdown(light_css, unsafe_allow_html=True)


st.title("Fake Seller Caption Detection 💢")
st.caption("AI model that detects fake seller captions from online marketplaces.")

# Load model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Load dataset for charts
df = pd.read_csv("data/sellercaption.csv")

# SIDEBAR - DATASET PREVIEW
st.sidebar.header("   Dataset Preview")
st.sidebar.dataframe(df.head(20))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub(r"[^a-zA-Z ]","",text)
    return text

# SESSION STATE FOR TEXTBOX
if "caption_text" not in st.session_state:
    st.session_state.caption_text = "Secret formula companies don't want you to know about! maybe"


def clear_text():
    st.session_state.caption_text = ""
    
user_input = st.text_area(
    "Enter seller caption:",
    key="caption_text"
)

col1, col2 = st.columns([1,1])

with col1:
    if st.button("Predict"):
        processed = clean_text(user_input)

        vec = vectorizer.transform([processed])

        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        confidence = max(prob) * 100

        if prediction == 1:
           st.success("🟢 Real Caption")
           st.write(f"Confidence: {confidence:.2f}%")
           st.progress(int(confidence))
        else:
           st.error("🔴 Fake Caption")
           st.write(f"Confidence: {confidence:.2f}%")
           st.progress(int(confidence))

with col2:
    st.button("Clear", on_click=clear_text)

# WordCloud
st.subheader(" WordCloud of Captions")

# Join all captions
text = " ".join(df["caption"].astype(str))

# Generate wordcloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="black"
).generate(text)

# Plot
fig_wc, ax_wc = plt.subplots()
ax_wc.imshow(wordcloud, interpolation="bilinear")
ax_wc.axis("off")

st.pyplot(fig_wc)


# DISTRIBUTION CHART
st.subheader(" Caption Label Distribution")

label_counts = df['label'].value_counts()

fig1, ax1 = plt.subplots()
sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax1)

ax1.set_xlabel("Label")
ax1.set_ylabel("Count")
ax1.set_title("Fake vs Real Caption Distribution")

st.pyplot(fig1)


# MODEL PERFORMANCE
st.subheader(" Model Performance")

# Clean label column
df['label'] = df['label'].astype(str).str.strip().str.lower()

# Map labels
df['label'] = df['label'].map({'fake':0,'real':1})

# Drop rows that became NaN after mapping
df = df.dropna(subset=['label'])

# Clean captions
df['caption'] = df['caption'].apply(clean_text)

X = df['caption']
y = df['label']

# Vectorize
X_vec = vectorizer.transform(X)

# Predict
y_pred = model.predict(X_vec)

# Metrics
accuracy = accuracy_score(y,y_pred)
precision = precision_score(y,y_pred)
recall = recall_score(y,y_pred)
f1 = f1_score(y,y_pred)

st.write("Accuracy:",accuracy)
st.write("Precision:",precision)
st.write("Recall:",recall)
st.write("F1 Score:",f1)


# CONFUSION MATRIX
st.subheader(" Confusion Matrix")

cm = confusion_matrix(y,y_pred)

fig2, ax2 = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax2)

st.pyplot(fig2)


# ROC CURVE
st.subheader(" ROC Curve")

y_prob = model.predict_proba(X_vec)[:,1]

fpr, tpr, _ = roc_curve(y,y_prob)
roc_auc = auc(fpr,tpr)

fig3, ax3 = plt.subplots()

ax3.plot(fpr,tpr,label="AUC = %0.2f" % roc_auc)
ax3.plot([0,1],[0,1],'--')

ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curve")
ax3.legend()

st.pyplot(fig3)