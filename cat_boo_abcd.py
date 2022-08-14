import pickle
import streamlit as st
import numpy as np

model_cat=pickle.load(open('cat.pkl','rb'))

def catb1(Hospital_code, Hospital_type_code, City_Code_Hospital, Hospital_region_code , Available_Extra_Rooms_in_Hospital, Department, Ward_Type, Ward_Facility_Code, Bed_Grade, City_Code_Patient, Type_of_Admission,Severity_of_Illness, Visitors_with_Patient, Age, Admission_Deposit):
    new_data = np.array([Hospital_code, Hospital_type_code, City_Code_Hospital, Hospital_region_code , Available_Extra_Rooms_in_Hospital, Department, Ward_Type, Ward_Facility_Code, Bed_Grade, City_Code_Patient, Type_of_Admission,Severity_of_Illness, Visitors_with_Patient, Age, Admission_Deposit])
    pred = model_cat.predict(new_data)
    proba = model_cat.predict_proba(new_data)
    print("Prediction Probability is " + str(proba))
    return pred
    
    
def main():
    st.title("Hospital Stay Prediction")
    html_temp = """
    <div style = "background-color:#025246; padding:10px">
    <h2 style = "color:white;text-align:center;">Predicting Length of Stay of a Patient in Hospital Based on User Input </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    Hospital_code = st.slider("Choose Hospital Code",1,32)
    
    Age=st.selectbox("Select Age", ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100'])
    dict_age={'0-10':0, '11-20':1, '21-30':2, '31-40':3, '41-50':4, '51-60':5, '61-70':6, '71-80':7, '81-90':8, '91-100':9}
    Age = dict_age[Age]
    
    Hospital_type_code = st.selectbox("Select Hospital type code", ['a','b','c','d','e','f','g'])
    dict_Hospital_type_code={'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6}
    Hospital_type_code = dict_Hospital_type_code[Hospital_type_code]

    Hospital_region_code=st.selectbox("Select Hospital region code", ['X','Y','Z'])
    dict_Hospital_region_code={'X':0, 'Y':1, 'Z':2}
    Hospital_region_code = dict_Hospital_region_code[Hospital_region_code]

    Department=st.selectbox("Select Department", ['surgery', 'TB & Chest disease', 'radiotherapy','anesthesia','gynecology'])
    dict_Department={'surgery':1, 'TB & Chest disease':2, 'radiotherapy':3,'anesthesia':4,'gynecology':5}
    Department = dict_Department[Department]

    Type_of_Admission=st.selectbox("Select Type of Admission", ['Emergency', 'Trauma', 'Urgent'])
    dict_Type_of_Admission={'Emergency':3, 'Trauma':1, 'Urgent':2}
    Type_of_Admission = dict_Type_of_Admission[Type_of_Admission]

    Ward_Facility_Code=st.selectbox("Select Ward_Facility_Code", ['A', 'B', 'C','D','E','F'])
    dict_Ward_Facility_Code={'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6}
    Ward_Facility_Code = dict_Ward_Facility_Code[Ward_Facility_Code]

    Ward_Type=st.selectbox("Select Ward Type", ['P', 'Q', 'R','S','T','U'])
    dict_Ward_Type={'P':1, 'Q':2, 'R':3,'S':4,'T':5,'U':6}
    Ward_Type = dict_Ward_Type[Ward_Type]

    Severity_of_Illness=st.selectbox("Select Severity of Illness", ["Minor", "Moderate", "Extreme"])
    dict_Severity_of_Illness={"Minor":0, "Moderate":1, "Extreme":2}
    Severity_of_Illness = dict_Severity_of_Illness[Severity_of_Illness]
    
    Bed_Grade= st.slider("Choose Bed Grade",1,4)

    City_Code_Hospital=st.selectbox("City Code Hospital", [1,2,3,4,5,6,7,9,10,11,13])
    
    Available_Extra_Rooms_in_Hospital=st.number_input("Available Extra Rooms in Hospital", min_value=0, step=1)
    City_Code_Patient=st.slider( "City Code Patient", 1,38)
    Visitors_with_Patient=st.number_input("Visitors with Patient", min_value=0, step=1)
    Admission_Deposit=st.number_input("Admission Deposit")
    
    more_html='''
    <div style = "background-color:#F08080; padding:10px">
    <h2 style = "color:white;text-align:center;"> You have to stay more than 10 days </h2>
    </div>
    '''
    
    less_html='''
    <div style = "background-color:#00FF00; padding:10px">
    <h2 style = "color:black;text-align:center;"> You have to stay less than 10 days </h2>
    </div>
    '''
    
    if st.button("Submit"):
        output = catb1(Hospital_code, Hospital_type_code, City_Code_Hospital, Hospital_region_code , Available_Extra_Rooms_in_Hospital, Department, Ward_Type, Ward_Facility_Code, Bed_Grade, City_Code_Patient, Type_of_Admission,Severity_of_Illness, Visitors_with_Patient, Age, Admission_Deposit)
        if(output==0):
            st.markdown(more_html, unsafe_allow_html=True)
        else:
            st.markdown(less_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
