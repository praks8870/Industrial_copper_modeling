import pandas as pd
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import pickle
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import re




st.set_page_config(page_title="Industrial Copper Modeling",
                   layout='wide',
                   initial_sidebar_state='expanded')

st.title("Industrial Copper Modeling")

st.markdown("""
    <style>
        .stApp {
            background-image: url("https://media.gettyimages.com/id/1274373899/photo/full-frame-side-of-a-newly-polished-copper-cooking-pot.jpg?s=612x612&w=0&k=20&c=M1rs5P-HpYjXpofg6DD6YYfw0caeejtEiQqiY65Gd18=");
            background-size: cover;
        }
        </style>
        """, unsafe_allow_html=True)

selected = option_menu(None, ['Predict Price', 'Predict Status'],
                       icons=[],
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": {"font-size": "35px", "text-align": "centre", "margin": "0px", "--hover-color": "#6495ED"},
                               "icon": {"font-size": "35px"},
                               "container" : {"max-width": "6000px"},
                               "nav-link-selected": {"background-color": "#6495ED"}})


if selected == 'Predict Price':

    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                     '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                     '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                     '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                     '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    
    with st.form("my_form"):
        col1,col2,col3=st.columns([5,2,5])
        with col1:
            st.write(' ')
            status = st.selectbox("Status", status_options,key=1)
            item_type = st.selectbox("Item Type", item_type_options,key=2)
            country = st.selectbox("Country", sorted(country_options),key=3)
            application = st.selectbox("Application", sorted(application_options),key=4)
            product_ref = st.selectbox("Product Reference", product,key=5)
        with col3:               
            st.write( f'<h5 style="color:[White];">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
            quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
            st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #009999;
                    color: white;
                    width: 100%;
                }
                </style>
            """, unsafe_allow_html=True)

        flag=0 
        pattern = "^(?:\d+|\d*\.\d+)$"

        for i in [quantity_tons,thickness,width,customer]:             
            if re.match(pattern, i):
                pass
            else:                    
                flag=1  
                break
    
        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)  
             
        if submit_button and flag==0:
            
            import pickle
            with open(r"D:\datascience\Copper_project\regression_model.pkl", 'rb') as f:
                regr = pickle.load(f)

            with open(r"D:\datascience\Copper_project\scaler_reg.pkl", 'rb') as f:
                scaler = pickle.load(f)

            with open(r"D:\datascience\Copper_project\enc_reg.pkl", 'rb') as f:
                ohe = pickle.load(f)

            with open(r"D:\datascience\Copper_project\end_reg2.pkl", 'rb') as f:
                ohe2 = pickle.load(f)


            test_data = np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
            td1 = test_data[:, [0,1,2,3,4,5,6]]
            td2 = ohe.transform(test_data[:, [7]]).toarray()
            td3 = ohe2.transform(test_data[:, [8]]).toarray()

            td4= np.concatenate((td1, td2, td3), axis=1)

            test_data = scaler.transform(td4)

            pred = regr.predict(test_data)[0]

            st.write('## :green[Predicted selling price:] ', np.exp(pred))


if selected == 'Predict Status':


    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                     '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                     '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                     '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                     '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

    with st.form("my_form1"):
        col1,col2,col3=st.columns([5,1,5])
        with col1:
            cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            cwidth = st.text_input("Enter width (Min:1, Max:2990)")
            ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
            cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
            
        with col3:    
            st.write(' ')
            citem_type = st.selectbox("Item Type", item_type_options,key=21)
            ccountry = st.selectbox("Country", sorted(country_options),key=31)
            capplication = st.selectbox("Application", sorted(application_options),key=41)  
            cproduct_ref = st.selectbox("Product Reference", product,key=51)           
            csubmit_button = st.form_submit_button(label="PREDICT STATUS")

        cflag=0 
        pattern = "^(?:\d+|\d*\.\d+)$"
        for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:             
            if re.match(pattern, k):
                pass
            else:                    
                cflag=1  
                break
        
    if csubmit_button and cflag==1:
        if len(k)==0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ",k)  
            
    if csubmit_button and cflag==0:
        import pickle
        
        with open(r"D:\datascience\Copper_project\clasification_model.pkl", 'rb') as f:
            model = pickle.load(f)

        with open(r"D:\datascience\Copper_project\scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)

        with open(r"D:\datascience\Copper_project\encoder.pkl", 'rb') as f:
            oh = pickle.load(f)

        test_data = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(cproduct_ref),citem_type]])

        test_data_numeric = np.array(test_data[:, [0, 1, 2, 3, 4, 5, 6, 7]], dtype=float)

        test_data_categorical = pd.DataFrame(test_data[:, [8]], columns=['item type'])

        test_data_oh = oh.transform(test_data_categorical).toarray()

        test_data_combined = np.concatenate((test_data_numeric, test_data_oh), axis=1)

        tdcs = scaler.transform(test_data_combined)

        cpred = model.predict(tdcs)

        if cpred==1:
            st.write('## :green[The Status is Won] ')
        else:
            st.write('## :red[The status is Lost] ')
