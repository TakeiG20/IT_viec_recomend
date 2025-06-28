import re
import string
import streamlit as st
import pandas as pd
import pickle
from gensim import corpora, models, similarities
from sklearn.metrics.pairwise import cosine_similarity


#------- Tạo công thức cho xuất đầu ra -----

# Đọc dữ liệu
df = pd.read_excel('Overview_Companies.xlsx')

#Load file 
#LOAD TEENCODE
file = open('teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
#LOAD STOPWORDS
file = open('vietnamese-stopwords_rev.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

#Hàm làm sạch dữ liệu
def clean_text(text):
    text = text.lower() #Hoa -> thường
    text = re.sub(rf"[{string.punctuation}]", "", text) #Bỏ dấu câu
    text = re.sub(r"\b(g|ml)\b", "", text)  # Bỏ từ 'g' hoặc 'ml' khi nó là cả từ
    text = re.sub(r"\s+", " ", text).strip()  # Xóa khoảng trắng thừa
    text = re.sub(r"^[\-\+\*\•\●\·\~\–\—\>]+", "", text) # Bỏ dấu đầu câu
    return text

def fix_teencode(text, mapping):
    words = text.split()
    corrected = [mapping.get(word, word) for word in words]
    return " ".join(corrected)


def remove_stopword(text, stopword_dict):
    words = text.split()
    stopword = [word for word in words if word not in stopword_dict]
    return " ".join(stopword)

def clean_pipeline(text):
    text = clean_text(text)
    # text = safe_translate_vi(text)
    text = fix_teencode(text, teen_dict)
    # text = remove_wrongword(text, wrong_lst)
    text = remove_stopword(text, stopwords_lst)
    return text


# Load dùng cho gensim
#Bài toán 1
with open("gensim_dictionary.pkl", "rb") as f:
    gensim_dictionary = pickle.load(f)
with open("gensim_corpus.pkl", "rb") as f:
    gensim_corpus = pickle.load(f)
gensim_tfidf = models.TfidfModel.load("gensim_tfidf_model.tfidf")
gensim_index = similarities.SparseMatrixSimilarity.load("gensim_similarity.index")
#Bài toán 2
with open("gensim_dictionary_2.pkl", "rb") as f:
    gensim_dictionary_2 = pickle.load(f)
gensim_tfidf_2 = models.TfidfModel.load("gensim_tfidf_model_2.tfidf")
gensim_index_2 = similarities.SparseMatrixSimilarity.load("gensim_similarity_2.index")

#Load dùng cho cosin
#Bài toán 1
with open("cosine_index.pkl", "rb") as f:
    cosine_index = pickle.load(f)
#Bài toán 2
with open("cosine_tfidf_2.pkl", "rb") as f:
    cosine_tfidf_2 = pickle.load(f)
with open("cosine_tfidf_matrix_2.pkl", "rb") as f:
    cosine_tfidf_matrix_2 = pickle.load(f)

# Tạo đầu ra cho gensim
# Bài toán 1
def find_similar_companies_gem(company_id, corpus, tfidf, index, top_n):
    # 1. Lấy vector TF-IDF của công ty được chọn
    tfidf_vec = tfidf[corpus[company_id]]
    # 2. Tính cosine similarity giữa công ty này và tất cả các công ty khác
    sims = index[tfidf_vec]  # (Là tính cosin giữa vector hỏi và matran vector đang có, cosin gần 1 thì càng // hay trùng nhau: [a, b] x [b, 1] = [a, 1])
    # 3. Sắp xếp theo độ tương tự giảm dần, loại chính nó ra, lấy top 5
    top_similar_gem_find = sorted([(i, sim) for i, sim in enumerate(sims) if i != company_id],key=lambda x: x[1],reverse=True)[:top_n]
    # 4. Lấy ID và similarity
    company_ids = [i[0] for i in top_similar_gem_find]
    similarities = [round(i[1], 4) for i in top_similar_gem_find]
    # 5. Lấy dữ liệu từ gốc
    df_gem_find = df.iloc[company_ids].copy()
    df_gem_find['similarity'] = similarities

    return df_gem_find, top_similar_gem_find, company_id

# Bài toán 2
def search_similar_companies_gem(query_text, clean_pipeline, dictionary, tfidf_model, index_2, data, top_n=1):
    # 1. Làm sạch và tách từ
    clean_text = clean_pipeline(query_text)
    tokens = clean_text.split()  # hoặc dùng tokenizer riêng nếu bạn có
    # 2. Chuyển sang dạng vector BoW
    bow_vector = dictionary.doc2bow(tokens)
    # 3. Chuyển sang vector TF-IDF
    tfidf_vector = tfidf_model[bow_vector]
    # 4. Tính độ tương tự với toàn bộ công ty
    sims = index_2[tfidf_vector]
    # 5. Sắp xếp theo độ tương tự giảm dần
    top_similar_gem_search = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_n]
    # 6. Lấy ID và similarity
    company_ids = [i[0] for i in top_similar_gem_search]
    similarities = [round(i[1], 4) for i in top_similar_gem_search]
    # 7. Lấy dữ liệu từ gốc
    df_gem_search = data.iloc[company_ids].copy()
    df_gem_search['similarity'] = similarities

    return df_gem_search, top_similar_gem_search, query_text

# Tạo đầu ra cho cosine-similarity
# Bài toán 1:
def find_similar_companies_cos(cosine_similarities, data, company_id, top_n):
    # Bỏ chính nó ra bằng cách gán -1
    sim_scores = cosine_similarities[company_id].copy()
    sim_scores[company_id] = -1
    # Lấy top_n chỉ số công ty tương tự nhất
    similar_indices = sim_scores.argsort()[-top_n:][::-1]
    # Tạo danh sách (score, index)
    top_similar_cos_find = [(i, sim_scores[i]) for i in similar_indices]
    # Lấy dòng dữ liệu công ty từ DataFrame
    df_cos_find = data.iloc[similar_indices].copy()
    df_cos_find["similarity"] = [sim_scores[i] for i in similar_indices]

    return top_similar_cos_find, df_cos_find, company_id
#Bài toán 2
def search_similar_companies_cos(query_text_2, vectorizer, tfidf_matrix, data, top_n=5):
    # 1. Làm sạch từ khóa truy vấn
    cleaned_query = clean_pipeline(query_text_2)
    # 2. Chuyển thành vector TF-IDF (dạng 1×n)
    query_vector = vectorizer.transform([cleaned_query])  # giữ nguyên từ điển cũ
    # 3. Tính độ tương đồng cosine với toàn bộ công ty
    sims = cosine_similarity(query_vector, tfidf_matrix)[0]  # kết quả là vector 1D
    # 4. Lấy top N công ty có điểm similarity cao nhất
    similar_indices = sims.argsort()[-top_n:][::-1]  # sắp xếp giảm dần
    # 5. Tạo kết quả danh sách điểm và chỉ số/
    top_similarity_cos_search = [(sims[i], i) for i in similar_indices]
    # 6. Tạo DataFrame các công ty tương tự
    df_cos_search = data.iloc[similar_indices].copy()
    df_cos_search["similarity_score"] = [sims[i] for i in similar_indices]

    return top_similarity_cos_search , df_cos_search, query_text_2

# Hàm lấy index từ danh sách công ty chọn
def suggest_company_name(df):
    # Tạo mapping: tên công ty → id (nếu tên trùng nhau sẽ lấy ID đầu tiên)
    company_mapping = (df.set_index("Company Name")["id"].to_dict())
    # Tạo danh sách tên duy nhất, đã sắp xếp
    company_list = sorted(company_mapping.keys())
    # Tạo selectbox
    selected_name = st.selectbox(
        "Chọn hoặc nhập tên công ty:",
        options=company_list
    )

    # Lấy id tương ứng
    selected_id = company_mapping.get(selected_name, None)
    return selected_name, selected_id

#Hàm hiển thị nội dung chi tiết công ty
def show_company_detail(data, title=None, expanded=False):
    with st.expander(title or f"{data['Company Name']} (ID: {data['id']})", expanded=expanded):
        st.markdown(f"""
                    • **Type:** {data['Company Type']}  
                    • **Industry:** {data['Company industry']}  
                    • **Size:** {data['Company size']}  
                    • **Country:** {data['Country']}  
                    • **Working days:** {data['Working days']}  
                    • **Overtime Policy:** {data['Overtime Policy']}  
                    • **Location:** {data['Location']}  
                    • **Link:** {data['Href']} 
                    """)
        st.markdown(f"**Overview:** {data['Company overview']}")
        st.markdown(f"**Key skills:** {data['Our key skills']}")
        st.markdown("**Why you'll love working here:** " + str(data["Why you'll love working here"]))

cols_show = ["id", "Company Name", "Company Type", "Company industry", "similarity"]

#------- Nội dung hiển thị trên tab -----
st.set_page_config(
  page_title="PROJECT_02",
  page_icon="  ",
  layout="wide",
  initial_sidebar_state="expanded",
) 

#------- Giao diện Streamlit -----
#Hình ảnh đầu tiên
st.image('images/channels4_banner.jpg', use_container_width=True)

# 3 tab nằm ngang
tab1, tab2, tab3 = st.tabs(["BUSINESS OVERVIEWS", "BUIL PROJECT", "NEW PREDICT"])

# Sidebar chứa dự án
with st.sidebar:
    st.sidebar.header("PROJECT_02")
    page = st.radio("Chọn nội dung:", ["GENSIM", "COSINE-SIMILARITY"])

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.sidebar.header('INFORMATION')
    st.sidebar.write('Vo Minh Tri')
    st.sidebar.write('Email: trivm203@gmail.com')
    st.sidebar.write('Pham Thi Thu Thao')
    st.sidebar.write('Email: thaofpham@gmail.com')

#Nội dung cho từng tab
with tab1:
    st.header("GENSIM AND COSINE-SIMILARITY")
    st.write('''
            Dùng thuật toán Gensim và Cosine-similarity, giải quyết 2 bài toán:\n
            - Bài toán 1: chọn 1 công ty, đề xuất 5 công ty tương tự\n
            - Bài toán 2: dựa vào mô tả để đề xuất 1 công ty phù hợp nhất\n
            ''')

with tab2:

    if page == "GENSIM":
        st.header("GENSIM")
        st.markdown('#### Bài toán 1:')
        st.write('Dùng các nội dung phân loại (Company Type, Company industry, Company size,...) làm dữ liệu đầu vào')
        st.image('images/gen1_dau vao.png')
        st.write('Dùng gensim tạo từ điển dictionary và từ điển tần số từ corpus')
        st.image('images/gen1_dictionary.png')
        st.image('images/gen1_copus.png')
        st.write('Vector hóa bằng tf-idf để tạo ma trận thưa thớt')
        st.write('Lấy vector tf-idf của 1 công ty được chọn rồi tính tỉ số tương tự so với ma trận thưa')
        st.write('Sắp xếp và lấy top5')
        st.image('images/gen1_top5.png')
        st.image('images/gen1_top5_df.png')
        st.markdown('#### Bài toán 2:')
        st.write("Dùng các nội dung mô tả tự do (Company overview, Our key skills, Why you'll love working here) làm dữ liệu đầu vào")
        st.image('images/gen2_input.png')
        st.write('Các bước tạo từ điển và tf-idf tương tự')
        st.write('Từ khóa tìm kiếm sẽ được biến đổi thành vector và so sánh chỉ số tương tự')
        st.write('sắp xếp và lấy công ty tương đồng nhất')
        st.image('images/gen2_top1.png')

        

    elif page == "COSINE-SIMILARITY":
        st.header("COSINE-SIMILARITY")
        st.markdown('#### Bài toán 1:')
        st.write('Dùng các nội dung phân loại (Company Type, Company industry, Company size,...) làm dữ liệu đầu vào')
        st.image('images/gen1_dau vao.png')
        st.write('Vector hóa trực tiếp bằng tf-idf để tạo ma trận thưa thớt')
        st.write('Tính tỉ số tương tự toàn bộ ma trận thưa')
        st.write('Trực quan hóa các công ty có chỉ số tương tự >0.5')
        st.image('images/cos1_matran.png')
        st.write('Chọn 1 công ty, thuật toán sẽ lấy hàng ngang, sắp xếp và lấy top5')
        st.image('images/cos1_top5.png')
        st.image('images/cos1_top5_df.png')
        st.markdown('#### Bài toán 2:')
        st.write("Dùng các nội dung mô tả tự do (Company overview, Our key skills, Why you'll love working here) làm dữ liệu đầu vào")
        st.image('images/gen2_input.png')
        st.write('Các bước tạo tf-idf tương tự')
        st.write('Từ khóa tìm kiếm sẽ được biến đổi thành vector và so sánh chỉ số tương tự')
        st.write('sắp xếp và lấy công ty tương đồng nhất')
        st.image('images/cos2_top1.png')
    
with tab3:
    if page == "GENSIM":
        st.header("GENSIM")
        st.markdown('#### Bài toán 1:')
        #input
        selected_name, selected_id = suggest_company_name(df)
        if selected_id is not None and selected_id in df['id'].values:
            # PROCESS
            df_gem_find, top_similar_gem_find, selected_id = find_similar_companies_gem(
                company_id=selected_id,
                corpus=gensim_corpus,
                tfidf=gensim_tfidf,
                index=gensim_index,
                top_n=3
            )

            # OUTPUT
            st.subheader("🏢 Công ty đang tìm kiếm")
            show_company_detail(df[df['id'] == selected_id].iloc[0])

            st.subheader("🏙️ Các công ty tương tự")
            st.dataframe(df_gem_find[cols_show].style.format({"similarity": "{:.4f}"}))

            for idx, row in df_gem_find.iterrows():
                show_company_detail(
                    row,
                    title=f"{row['Company Name']} (Similarity: {row['similarity']:.4f})"
                )
        else:
            st.info("Vui lòng chọn một công ty để xem gợi ý tương tự.")
        
        st.markdown('#### Bài toán 2:')
        #input
        query_text=st.text_input('Nhập từ khóa: ')
        if query_text:
            # PROCESS
            df_gem_search, top_similar_gem_search, query_text = search_similar_companies_gem(
                query_text=query_text,
                clean_pipeline=clean_pipeline,
                dictionary=gensim_dictionary_2,
                tfidf_model=gensim_tfidf_2,
                index_2=gensim_index_2,
                data=df,
                top_n=1
            )

            # OUTPUT
            if df_gem_search is not None and not df_gem_search.empty:
                search_id, search_name = top_similar_gem_search[0]

                st.subheader("🏢 Công ty tương đồng nhất")
                st.dataframe(df_gem_search[cols_show].style.format({"similarity": "{:.4f}"}))

                show_company_detail(
                    df[df['id'] == search_id].iloc[0],
                    title=f"{df_gem_search.iloc[0]['Company Name']} (Similarity: {df_gem_search.iloc[0]['similarity']:.4f})",
                    expanded=True
                )
            else:
                st.warning("Không tìm thấy công ty tương đồng.")
       
    elif page == "COSINE-SIMILARITY":
        st.header("COSINE-SIMILARITY")
        st.markdown('#### Bài toán 1:')
        #input
        selected_name, selected_id = suggest_company_name(df)
        if selected_id is not None and selected_id in df['id'].values:
            #process
            top_similar_cos_find, df_cos_find, selected_id = find_similar_companies_cos(cosine_index, df, company_id=selected_id, top_n=3)
            #output
            st.subheader("🏢 Công ty đang tìm kiếm")
            show_company_detail(df[df['id'] == selected_id].iloc[0])

            st.subheader("🏙️ Các công ty tương tự")
            st.dataframe(df_cos_find[cols_show].style.format({"similarity": "{:.4f}"}))
            for idx, row in df_cos_find.iterrows():
                show_company_detail(
                    row,
                    title=f"{row['Company Name']} (Similarity: {row['similarity']:.4f})"
                )
        else:
            st.info("Vui lòng chọn một công ty để xem gợi ý tương tự.")
        

        st.markdown('#### Bài toán 2:')
        #input
        query_text_2 = st.text_input('Nhập bình luận của bạn: ')
        if query_text_2:
            #process
            top_similarity_cos_search , df_cos_search, query_text_2 = search_similar_companies_cos(query_text_2=query_text_2, vectorizer=cosine_tfidf_2, tfidf_matrix=cosine_tfidf_matrix_2, data=df, top_n=1)
            #output
            if df_cos_search is not None and not df_cos_search.empty:
                search_id, search_name = top_similarity_cos_search[0]

                st.subheader("🏢 Công ty tương đồng nhất")
                st.dataframe(df_cos_search[cols_show].style.format({"similarity": "{:.4f}"}))

                show_company_detail(
                    df[df['id'] == search_id].iloc[0],
                    title=f"{df_cos_search.iloc[0]['Company Name']} (Similarity: {df_cos_search.iloc[0]['similarity']:.4f})",
                    expanded=True
                )
            else:
                st.warning("Không tìm thấy công ty tương đồng.")
  
                                 
# Adding a footer
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
.footer p {
    font-size: 20px;  /* Bạn có thể chỉnh to hơn nếu muốn, ví dụ 28px */
    color: blue;
    margin: 10px 0;  /* Để chữ không dính sát mép footer */
}

</style>
<div class="footer">
<p> Trung tâm Tin Học - Trường Đại Học Khoa Học Tự Nhiên <br> Đồ án tốt nghiệp Data Science </p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)