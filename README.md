# fraud-detection-streamlit
	Real-Time Financial Fraud Detection System
  A complete end-to-end fraud detection system developed during our internship at ZIDIO. This solution uses a combination of supervised and unsupervised machine learning models, integrated with real-time monitoring, alerting, and visual dashboards to detect and flag fraudulent financial transactions.
Project Highlights
•	Achieved 99%+ accuracy using advanced machine learning algorithms (XGBoost, LightGBM, Random Forest).
•	Integrated real-time fraud prediction pipeline with Apache Kafka.
•	Developed an interactive Streamlit dashboard for live monitoring of transactions.
•	Enabled email alert notifications for suspicious/high-risk transactions.
•	Added unsupervised anomaly detection using Isolation Forest and PCA for novel fraud pattern detection.
•	Incorporated graph-based fraud ring visualization using NetworkX.
Tech Stack
•	Programming Language: Python
•	Libraries & ML Frameworks: Pandas, Scikit-learn, XGBoost, LightGBM
•	Real-Time Processing: Apache Kafka
•	Dashboard: Streamlit
•	Alert System: SMTP (email integration)
•	Graph Analysis: NetworkX
Models Implemented
Model	Type	Highlights
Random Forest	Supervised	High accuracy, robustness to overfitting
XGBoost	Supervised	Best overall model performance
LightGBM	Supervised	Fast training and scoring with high precision
Isolation Forest	Unsupervised	Detects anomalies not seen during training
PCA (Reconstruction Error)	Unsupervised	Captures irregular transactions via error size
Graph-Based Fraud Detection	Conceptual	Visualizes fraudulent clusters and connections

Real-Time Monitoring Dashboard
•	Built using Streamlit for a user-friendly experience.
•	Displays real-time logs of transactions processed.
•	Shows fraud vs. legit transaction counts in bar charts.
•	Dashboard auto-refreshes every 5 seconds for live updates.
•	Includes visual indicators of suspicious patterns and model decisions.
Alert System
•	Configured using SMTP to send real-time email alerts.
•	Automatically triggers on detection of fraudulent transactions.
•	Provides detailed info in alerts (amount, risk score, timestamp).
•	Ensures timely awareness and response for high-risk activity.
Future Enhancements
•	AI-driven risk scoring system
•	Blockchain-based transaction tracking
•	Mobile app for fraud alerts on the go
Conclusion
     This real-time financial fraud detection system represents a comprehensive and forward-thinking approach to combating digital financial crimes. By integrating both supervised and unsupervised machine learning models, the system is capable of detecting known fraud patterns and adapting to emerging, previously unseen threats.
      The use of Apache Kafka ensures seamless real-time data streaming, while Streamlit enables intuitive, real-time visualization for analysts and stakeholders. Additional features like automated email alerts and graph-based fraud ring detection (using NetworkX) enhance both the operational impact and analytical depth of the system.
       Developed during our internship at ZIDIO, this project demonstrates not just technical skills but also the real-world applicability of machine learning in critical domains like finance. Its modular, scalable design allows for easy integration with future technologies—such as AI-powered risk scoring or blockchain-based transaction validation.
      In essence, this project combines accuracy, efficiency, and real-time responsiveness to deliver a practical and robust solution to the growing challenge of financial fraud.
 
