import pandas as pd
import streamlit as st
import hydralit_components as hc
import requests
from streamlit_lottie import st_lottie
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px  
import streamlit_card as st_card
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from pathlib import Path
import base64
#######################################################
#make it look nice from the start
st.set_page_config(layout='wide' ,page_title= 'Money Laundering Detection',
page_icon= '💲', initial_sidebar_state= 'expanded')

def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <h2 style = "color:#010101; text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "color:#010101; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)


# specify the primary menu definition
menu_data = [
    {'icon': "fas fa-tachometer-alt", 'label':"Network Graph",'ttip':"eda"},
    {'icon': "fas fa-eye", 'label':"Connectivity and Purity",'ttip':"discover"},
    {'icon': "fas fa-chart-line", 'label':"Dashboard",'ttip':"dash"},
    {'icon': "fas fa-robot", 'label':"AML Tool"}
]

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes("faded.png")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)



#html= """<div id="root"><div><style>:root {--menu_background: #357af7;--txc_inactive: #f0f2f6;--txc_active:#31333F;--option_active:#ffffff;}</style><nav class="navbar navbar-expand-custom navbar-mainbg w-100 py-0 py-md-0"><button class="navbar-toggler" type="button" aria-expanded="false"><i class="fas fa-bars text-white"></i></button><div class="navbar-collapse" id="complexnavbarSupportedContent" style="display: none;"><ul class="navbar-nav py-0"><div class="hori-selector" style="top: 0px; left: 203.683px; height: 56px; width: 129.5px;"><div class="left"></div><div class="right"></div></div><li class="nav-item py-0"><a class="nav-link" href="#0" data-toggle="tooltip" data-placement="top" data-html="true" title="Home"><i class="fa fa-home"></i>  Home</a></li><li class="nav-item py-0"><a class="nav-link" href="#1" data-toggle="tooltip" data-placement="top" data-html="true" title="eda"><i class="fas fa-tachometer-alt"></i> Explore</a></li><li class="nav-item py-0 active"><a class="nav-link" href="#2" data-toggle="tooltip" data-placement="top" data-html="true" title="dash"><i class="fas fa-chart-line"></i> Dashboard</a></li><li class="nav-item py-0"><a class="nav-link" href="#3" data-toggle="tooltip" data-placement="top" data-html="true" title="discover"><i class="fas fa-eye"></i> Discover</a></li><li class="nav-item py-0"><a class="nav-link" href="#4" data-toggle="tooltip" data-placement="top" data-html="true"><i class="fas fa-robot"></i> Machine Learning</a></li></ul></div>
#<img src="https://valoores.com/images/logo.png" style="float:left" width="400" height="100"></nav></div></div>"""
#st.markdown(html, unsafe_allow_html=True)


over_theme = {'menu_background':'#357af7'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home', #will show the st hamburger as well as the navbar now!
    sticky_nav=False, #at the top or not
    hide_streamlit_markers=False,
    sticky_mode='sticky', #jumpy or not-jumpy, but sticky or pinned
)

def load_lottieurl(url):
    r= requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie= load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_6f3b5s8z.json')
#get the id of the menu item clicked

if menu_id== 'Home':
    left, right= st.columns((1,1))
    with left:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        display_app_header(main_txt='Money Laundering Detection',
                       sub_txt='Notwithstanding, regardless of various expected applications inside the financial services sector, explicitly inside the Anti-Money Laundering (AML) appropriation of Artificial Intelligence and Machine Learning (ML) has been generally moderate.')
    
        display_app_header(main_txt='Motivation',
        sub_txt= 'For banks, now is the ideal opportunity to deploy ML models into their ecosystem. Despite this opportunity, increased knowledge and the number of ML implementations prompted a discussion about the feasibility of these solutions and the degree to which ML should be trusted and potentially replace human analysis and decision-making. In order to further exploit and achieve ML promise, banks need to continue to expand on its awareness of ML strengths, risks, and limitations and, most critically, to create an ethical system by which the production and use of ML can be controlled and the feasibility and effect of these emerging models proven and eventually trusted.')
    with right:
        st_lottie(lottie, height= 600,key= 'coding')
##############################################################################################
import functions
from pickle import TRUE
import pandas as pd
import datetime
df= pd.read_csv("ML1.csv")
dff= pd.read_csv("ML.csv")
df['Dates'] = pd.to_datetime(df['Date']).dt.date
df.Dates = pd.to_datetime(df.Dates)
df["Month"] = df.Dates.dt.month
df['Time'] = pd.to_datetime(df['Date']).dt.time
df.drop('Date', inplace=True, axis=1)
df['Hour']= df['Time'].apply(lambda x: x.hour)
df['Day']= df['Dates'].apply(lambda x: x.day)
df.drop('Dates', inplace=True, axis=1)
df.drop('Time', inplace=True, axis=1)
df['First Digit of Amount Transferred'] = df['Amount of Money Transferred'].astype(str).str[:1].astype(int)
df['Second Digit of Amount Transferred'] = df['Amount of Money Transferred'].astype(str).str[1:2].astype
df['Connectivity of Sender ID'] = df.groupby('Sender ID')['Sender ID'].transform('size')/len(df) *100
df['Connectivity of Receiver ID'] = df.groupby('Receiver ID')['Receiver ID'].transform('size')/len(df) *100
df['Purity of Sender ID']= 1- df['Connectivity of Sender ID']
df['Purity of Receiver ID']= 1- df['Connectivity of Receiver ID']

df.drop('Type of Fraud', axis=1, inplace=True) 

X= df.loc[:, ~df.columns.isin(['Is Fraud', 'Sender ID', 'Receiver ID'])]
y= df['Is Fraud'].astype('category')

num_features= X.select_dtypes(include= ['int64', 'int32', 'float64'])
cat_features= X.select_dtypes(include= ['object', 'category'])

if menu_id== 'Network Graph':
    display_app_header(main_txt='Connection Networks for a Sample Fraudulent and Non-Fraudulant Transactions During one Month',
                       sub_txt='')
    col1, col2= st.columns([1,1])
    with col1:
        st.image("https://i.postimg.cc/mgNr9Nsq/Figure-2.png")
        st.markdown("""
Most of the senders who committed money laundering transactions 
appeared to send the money to a person only one time, without any repetitive transactions to the 
same person. As if every time a money laundering transaction occurs a new false identification is created to 
receive the money/transaction and customers attempt to hide the size of a large cash transaction by breaking it 
into multiple, smaller transactions by conducting smaller transfer.
In addition, the amount of money transferred from senders to 
receivers is the same which raises suspicions around these transactions.""")
    with col2:
        st.image("https://i.postimg.cc/gjNw3V8m/Figure-1.png")
        st.markdown(""" On the contrary, safe transactions had more connectivity between senders and receivers, and the amount being transfered was different every time.
        """)
################################################################################################3
if menu_id== 'Dashboard':

    col1, col2= st.columns([8,8])
    with col1:
        
        wch_colour_box = (217,229,255)
        wch_colour_font = (0,0,0)
        fontsize = 30
        valign = "left"
        iconname = "fas fa-asterisk"
        sline = "Fraud Transaction Percentage"
        #lnk = '<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.12.1/css/all.css" crossorigin="anonymous">'
        i = "59%"

        htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                    {wch_colour_box[1]}, 
                                                    {wch_colour_box[2]}, 1); 
                            color: rgb({wch_colour_font[0]}, 
                                        {wch_colour_font[1]}, 
                                        {wch_colour_font[2]}, 0.75); 
                            font-size: {fontsize}px; 
                            border-radius: 7px; 
                            padding-left: 12px; 
                            padding-top: 18px; 
                            padding-bottom: 18px; 
                            line-height:25px;'>
                            <i class='{iconname} fa-xs'></i> {i}
                            </style><BR><span style='font-size: 14px; 
                            margin-top: 0;'>{sline}</style></span></p>"""

        st.markdown( htmlstr, unsafe_allow_html=True)
    with col2:

        wch_colour_font = (0,0,0)
        sline = "Safe Transaction Percentage"
        #lnk = '<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.12.1/css/all.css" crossorigin="anonymous">'
        i = "41%"

        htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                    {wch_colour_box[1]}, 
                                                    {wch_colour_box[2]}, 1); 
                            color: rgb({wch_colour_font[0]}, 
                                        {wch_colour_font[1]}, 
                                        {wch_colour_font[2]}, 0.75); 
                            font-size: {fontsize}px; 
                            border-radius: 7px; 
                            padding-left: 12px; 
                            padding-top: 18px; 
                            padding-bottom: 18px; 
                            line-height:25px;'>
                            <i class='{iconname} fa-xs'></i> {i}
                            </style><BR><span style='font-size: 14px; 
                            margin-top: 0;'>{sline}</style></span></p>"""

        st.markdown( htmlstr, unsafe_allow_html=True)
    
    only1= df[df['Is Fraud']==1]
    col1, col2= st.columns([8,8])
    #figure 0
    fig0 = px.pie(only1, names="Type of Transaction",  
             color="Type of Transaction",
             color_discrete_map={ # replaces default color mapping by value
                'cash-in': "#c5cfef", 'transfer': "#644BFF"
            })
    fig0.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)'})
    fig0.update_layout(title= 'Type of Transaction in Fraud Transactions')
    fig0.update_layout(yaxis_title= None)
    col1.plotly_chart(fig0, use_container_width= True)

    #figure 1
    only0= dff[dff['Is Fraud']==0]
    fig1 = px.pie(only0, names="Type of Transaction", 
             
             color="Type of Transaction",
             color_discrete_map={ # replaces default color mapping by value
                'cash-in': "#c5cfef", 'transfer': "#644BFF"
            })
    fig1.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)'})
    fig1.update_layout(title= 'Type of Transaction in Safe Transactions')
    fig1.update_layout(yaxis_title= None)
    col2.plotly_chart(fig1, use_container_width= True)

    #fig2
    fig2= px.histogram(df, x="Month", 
                color='Is Fraud', 
            barmode='group',
            height=400,
            color_discrete_map={ # replaces default color mapping by value
            0: "#c5cfef", 1: "#644BFF"
        })
    fig2.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)'})
    fig2.update_layout(title= 'Month vs Money Laundering Detection')
    fig2.update_layout(yaxis_title= None)
    col1.plotly_chart(fig2, use_container_width= True)

    #fig3
    fig3= px.histogram(df, x="Day of the Week", y= 'Amount of Money Transferred', 
                color='Is Fraud', 
            barmode='group',
            height=400,
            color_discrete_map={ # replaces default color mapping by value
            0: "#c5cfef", 1: "#644BFF"
        })

    fig3.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)'})
    fig3.update_layout(title= 'Day of the Week vs Money Laundering Detection')
    fig3.update_layout(yaxis_title= None)
    col2.plotly_chart(fig3, use_container_width= True)

    #fig4
    df_grouped = (
        only1.groupby(only1['Day']
        )['Is Fraud'].count().rename('Count').to_frame()
    )
    df_grouped.reset_index(inplace=True)
    
    fig4= px.line(df_grouped, x="Day", y='Count')
    fig4.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)'})
    fig4.update_layout(title= 'Day vs Money Laundering Detection')
    fig4.update_layout(yaxis_title= None)
    col1.plotly_chart(fig4, use_container_width= True)


    #fig5
    df_grouped1 = (
        only1.groupby(only1['Hour']
        )['Is Fraud'].count().rename('Count').to_frame()
    )
    df_grouped1.reset_index(inplace=True)
    fig5= px.line(df_grouped1, x= 'Hour', y= 'Count')
    fig5.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)'})
    fig5.update_layout(title= 'Day-Time vs Money Laundering Detection')
    fig5.update_layout(yaxis_title= None)
    col2.plotly_chart(fig5, use_container_width= True)

##########################################################################
if menu_id== 'Connectivity and Purity':

    connectivity_receiver= pd.read_csv("AVG Connectivity of Receiver ID.csv")
    connectivity_sender= pd.read_csv("AVG Connectivity of Sender ID.csv")
    purity_receiver= pd.read_csv("AVG Purity of Receiver ID.csv")
    purity_sender= pd.read_csv("AVG Purity of Sender ID.csv")

    one = px.bar(connectivity_receiver, y="Avg Connectivity of Receiver ID",  x='Is Fraud', color_discrete_map={0: "#644BFF", 1: "#644BFF"})
    one.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)'})
    one.update_layout(yaxis_range=[0,0.9])
    one.update_layout(xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
        ), title= 'Average Connectivity of Receiver ID')
    one.update_layout(yaxis_title= None)

    two= px.bar(connectivity_sender, y= "Avg Connectivity of Sender ID", x= 'Is Fraud', color_discrete_map={0: "#644BFF", 1: "#644BFF"})
    two.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)'})
    two.update_layout(yaxis_range=[0,0.9])
    two.update_layout(xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
        ),title= 'Average Connectivity of Sender ID')
    two.update_layout(yaxis_title= None)

    three= px.bar(purity_receiver, y= "Avg Purity of Receiver ID", x= 'Is Fraud')
    three.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)'})
    three.update_layout(yaxis_range=[0,0.9])
    three.update_layout(xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
        ),title= 'Average Purity of Receiver ID')
    three.update_layout(yaxis_title= None)

    four= px.bar(purity_sender, y= "Avg Purity of Sender ID", x= 'Is Fraud')
    four.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)'})
    four.update_layout(yaxis_range=[0,0.9])
    four.update_layout(xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
        ),title= 'Average Purity of Sender ID')
    four.update_layout(yaxis_title= None)

    col1, col2= st.columns([8,8])
    col1.plotly_chart(one, use_container_width= True)
    col2.plotly_chart(two, use_container_width= True)
    col1.plotly_chart(three, use_container_width= True)
    col2.plotly_chart(four, use_container_width= True)
############################################################
if menu_id== 'AML Tool':
    display_app_header(main_txt='Select Features and Get Transaction Status',
                       sub_txt='')
    sender= (
        df.groupby(df['Sender ID']
        ).agg({'Connectivity of Sender ID':'mean', 'Purity of Sender ID':'mean'})
    )    
    sender.reset_index(inplace=True)
    
    receiver= (
        df.groupby(df['Receiver ID']
        ).agg({'Connectivity of Receiver ID':'mean', 'Purity of Receiver ID':'mean'})
    )    
    receiver.reset_index(inplace=True)

    c1, c2 = st.columns([ 1, 1])
    with c1:
        z=st.number_input('Sender ID', 0, 9999999)
        if z in list(sender['Sender ID']):
            a=st.write('Connectivity of this sender is: ', sender[sender['Sender ID']==z]['Connectivity of Sender ID'].values[0])
            b=st.write('Purity of this sender is: ', sender[sender['Sender ID']==z]['Purity of Sender ID'].values[0])
        else:
            st.write('Connectivity of this sender is: 0')
            st.write('Purity of this sender is: 100')
    with c2:
        x= st.number_input('Receiver ID', 0, 9999999)
        if x in list(receiver['Receiver ID']):
            c=st.write('Connectivity of this receiver is: ', receiver[receiver['Receiver ID']==x]['Connectivity of Receiver ID'].values[0])
            d= st.write('Purity of this receiver is: ', receiver[receiver['Receiver ID']==x]['Purity of Receiver ID'].values[0])
        else:
            st.write('Connectivity of this receiver is: 0')
            st.write('Purity of this receiver is: 100')

    #Pipeline for numerical features preprocessing (imputing by mean, outliers removal using RobustScaler(), scaling and normalizing)
    numerical_transformer = Pipeline(steps=[                            
    ('outliers_removal',RobustScaler(with_centering=False,with_scaling=True)),
    ('num_imputer', SimpleImputer(missing_values = np.nan, strategy='mean'))])

    #Pipeline for categorical features preprocessing (imputing missing values by mode, encoding using OHE Dummies)
    categorical_transformer = Pipeline(steps=[                            
    ('cat_imputer', SimpleImputer(missing_values = np.nan, strategy='most_frequent')),
    ('encoder',OneHotEncoder(drop='first',handle_unknown='ignore',sparse=False))
    ])

    #Fitting the numerical and categorical features datasets into their corresponding transformers
    transformer = ColumnTransformer( transformers=[
        ('numerical', numerical_transformer, num_features.columns),
        ('categorical',categorical_transformer,cat_features.columns)]
        ,remainder='passthrough')  
    #############################
    def input_features():
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            typeoftx= st.selectbox('Type of Transaction',['cash-in', 'transfer'] )
            amount= st.number_input('Amount of Money Transferred', 0, 2000000000000)
            month= st.number_input('Month', 1, 12)
            dayofweek= st.selectbox('Day of the Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        with c2:
            hour= st.number_input('Hour', 0, 24)
            day= st.number_input('Day', 1, 31)
            firstdig= st.number_input('First Digit of Amountof Money Transferred', 1, 9)
            seconddig= st.number_input('Second Digit of Amountof Money Transferred', 1, 9)
        with c3:
            connectivity_sender= st.number_input('Connectivity of the Sender', step=1.,format='%.2f')
            purity_sender= st.number_input('Purity of the Sender', step=1.,format='%.2f')
            connectivity_receiver= st.number_input('Connectivity of the Receiver', step=1.,format='%.2f')
            purity_receiver= st.number_input('Purity of the Receiver', step=1.,format='%.2f')
        
        data= {'Type of Transaction': typeoftx, 'Amount of Money Transferred': amount,'Day of the Week': dayofweek, 
        'Month':month,'Hour':hour, 'Day':day, 'First Digit of Amount Transferred': firstdig, 'Second Digit of Amount Transferred': seconddig,
        'Connectivity of Sender ID': connectivity_sender, 'Purity of Sender ID': purity_sender,
        'Connectivity of Receiver ID': connectivity_receiver, 'Purity of Receiver ID': purity_receiver}

        features= pd.DataFrame(data, index= [0])
        return features


    df1= input_features()
    from sklearn import preprocessing
    encoder= preprocessing.LabelEncoder()
    y= encoder.fit_transform(y)
    from sklearn.ensemble import RandomForestClassifier
    model= RandomForestClassifier()
    pipeline= Pipeline(steps= [('transformer', transformer) , ('model', model)])
    pipeline.fit(X, y)
    pred= pipeline.predict(df1)

#accuracy of hyda
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=77)
    pipeline.fit(X_train, y_train)
    y_pred= pipeline.predict(X_test)
    acc= accuracy_score(y_test, y_pred)

    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #357af7;
        color:#ffffff; width: 90em;
    }
    div.stButton > button:hover {
        background-color: #5D95F9;
        width: 110em;color:#fffff;
        }
    </style>""", unsafe_allow_html=True)

    
    if st.button('Predict'):
        if pred==1:
            st.error('Suspicious Transaction')
        else:
            st.success('Safe Transaction')
        st.text(f"The Output's Accuracy is: {acc}")

    
