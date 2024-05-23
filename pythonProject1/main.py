import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN

import streamlit as st

st.title("Εφαρμογή Ανάλυσης Δεδομένων", anchor='center')
uploaded_file = st.file_uploader("Φορτώστε ένα αρχείο CSV ή Excel", type=["csv", "xlsx"])


def show_info():
    st.title("Εφαρμογή Ανάλυσης Δεδομένων", anchor='center')
    st.header("Πληροφορίες Εφαρμογής")

    st.subheader("Σχετικά με την Εφαρμογή")
    st.write("Η εφαρμογή αναλύει δεδομένα από την βάση δεδομένων iris.")

    st.subheader("Ομάδα Ανάπτυξης")
    st.write("Γεώργιος Καστρινός <span style='color:green'>(inf2021083)</span>", unsafe_allow_html=True)
    st.write("Στράτος Παπαδόπουλος <span style='color:blue'>(inf2021173)</span>", unsafe_allow_html=True)


if __name__ == "__main__":
    show_info()

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    st.write(df)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.iloc[:, :-1])

    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(df.iloc[:, :-1])

    st.subheader('PCA Visualization')
    fig_pca = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1])
    st.plotly_chart(fig_pca)

    st.subheader('t-SNE Visualization')
    fig_tsne = px.scatter(x=tsne_result[:, 0], y=tsne_result[:, 1])
    st.plotly_chart(fig_tsne)

    st.subheader('Histograms')
    for column in df.columns[:-1]:
        fig_hist = px.histogram(df, x=column)
        st.plotly_chart(fig_hist)

    st.subheader('Scatter Plot')
    x_feature = st.selectbox('Select X-axis feature', df.columns[:-1])
    y_feature = st.selectbox('Select Y-axis feature', df.columns[:-1])
    fig_scatter = px.scatter(df, x=x_feature, y=y_feature)
    st.plotly_chart(fig_scatter)

    tabs = st.tabs(["Κατηγοριοποίηση", "Ομαδοποίηση"])

    if tabs[0]:
        st.write("Επιλέξτε αλγόριθμο κατηγοριοποίησης:")
        classification_algorithm = st.selectbox('Επιλέξτε αλγόριθμο', ['k-NN', 'k-Means'])
        if classification_algorithm == 'k-NN':
            k = st.slider('Επιλέξτε την παράμετρο k', min_value=1, max_value=10, value=5)

            classifier = KNeighborsClassifier(n_neighbors=k)
            classifier.fit(df.iloc[:, :-1], df.iloc[:, -1])
            predictions = classifier.predict(df.iloc[:, :-1])
            accuracy = accuracy_score(df.iloc[:, -1], predictions)
            precision = precision_score(df.iloc[:, -1], predictions, average='weighted')
            recall = recall_score(df.iloc[:, -1], predictions, average='weighted')
            f1 = f1_score(df.iloc[:, -1], predictions, average='weighted')
            st.write("Classification Algorithm: k-NN")
            st.write("Parameters: k =", k)
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision)
            st.write("Recall:", recall)
            st.write("F1 Score:", f1)

        elif classification_algorithm == 'k-Means':
            k_clusters = st.slider('Επιλέξτε τον αριθμό των clusters', min_value=2, max_value=10, value=3)

            clusterer = KMeans(n_clusters=k_clusters)
            clusterer.fit(df.iloc[:, :-1])
            inertia = clusterer.inertia_
            st.write("Classification Algorithm: k-Means")
            st.write("Parameters: Number of clusters =", k_clusters)
            st.write("Inertia:", inertia)


    elif tabs[1]:
        st.write("Επιλέξτε αλγόριθμο ομαδοποίησης:")
        clustering_algorithm = st.selectbox('Επιλέξτε αλγόριθμο', ['k-Means', 'DBSCAN'])
        if clustering_algorithm == 'k-Means':
            k_clusters = st.slider('Επιλέξτε τον αριθμό των clusters', min_value=2, max_value=10, value=3)

            clusterer = KMeans(n_clusters=k_clusters)
            clusterer.fit(df.iloc[:, :-1])
            inertia = clusterer.inertia_
            st.write("Clustering Algorithm: k-Means")
            st.write("Parameters: Number of clusters =", k_clusters)
            st.write("Inertia:", inertia)
        elif clustering_algorithm == 'DBSCAN':
            epsilon = st.slider('Επιλέξτε την παράμετρο ε', min_value=0.1, max_value=10.0, value=1.0)

            clusterer = DBSCAN(eps=epsilon)
            clusterer.fit(df.iloc[:, :-1])
            st.write("Clustering Algorithm: DBSCAN")
            st.write("Parameters: ε =", epsilon)

















