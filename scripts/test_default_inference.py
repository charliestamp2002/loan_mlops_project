from loan_mlops.data.default_inference import predict_default_proba
from loan_mlops.data.approval_inference import predict_approval_proba

from loan_mlops.data.load_data import load_approval_data, load_default_data

df_approval = load_approval_data()
# print(df_approval.columns)

print("Three rows from approval data:")
approval_three_rows = df_approval.loc[0:2, [   
    "Age",
    "AnnualIncome",
    "CreditScore",
    "EmploymentStatus",
    "MaritalStatus",
    "EducationLevel"
    ]]

df_default = load_default_data()
# print(df_default.columns)
default_three_rows = df_default.loc[0:2, [ 
    "Age",
    "LoanAmount",
    "CreditScore", 
    "Education",
    "EmploymentType",
    "MaritalStatus"
    ]]

print(f"Three rows from approval data: \n{approval_three_rows}")
print(f"Three rows from default data: \n{default_three_rows}")

approval_example_applicant = {
   
   "Age": 35,
    "AnnualIncome": 75000,
    "CreditScore": 680,
    "EmploymentStatus": "Employed",
    "MaritalStatus": "Single",
    "EducationLevel": "Associate"
}
default_example_applicant = {
    "Age": 40,
    "LoanAmount": 15000,
    "CreditScore": 720,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married"
}

print(f"Probability prediction of one applicant (Approval): {predict_approval_proba(approval_example_applicant)}")
print(f"Probability prediction of one applicant (Approval): {predict_default_proba(default_example_applicant)}")