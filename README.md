# Flight Delay Analysis Project

This project analyzes a real-world flight dataset containing over 6 million records. The data includes detailed flight-level information such as scheduled and actual departure/arrival times, various delay types, taxi times, airline details, and origin/destination airports. The goal is to explore, visualize, and model the factors contributing to flight delays.

## Research Questions

To guide our analysis, we will address the following key research questions:

1. **Delay Propagation & Causality:**  
   - **Q:** How do departure delays impact arrival delays?  
   - **Objective:** Investigate whether longer departure delays lead to proportional increases in arrival delays and identify the intermediate factors (e.g., taxi out/in times) that contribute most.

2. **Airline Reliability & Performance:**  
   - **Q:** Which airlines consistently perform better in terms of on-time departures and arrivals?  
   - **Objective:** Compare performance across airlines by analyzing delay statistics, cancellation rates, and delay causes.

3. **Temporal Patterns & Seasonality:**  
   - **Q:** How do flight delays vary by time of day, day of the week, and month?  
   - **Objective:** Explore systematic delay patterns that may be linked to operational schedules or weather conditions.

4. **Airport & Route Analysis:**  
   - **Q:** Are there specific origin-destination pairs or hub airports that experience disproportionately high delays or cancellations?  
   - **Objective:** Identify hotspots of delay and evaluate the factors behind these route-specific challenges.

5. **Delay Attribution:**  
   - **Q:** What are the relative contributions of different delay causes (e.g., airline, weather, security, late aircraft) to the overall delay time?  
   - **Objective:** Break down the delay components to understand which factors are most critical for operational improvements.

6. **Predictive Modeling Feasibility:**  
   - **Q:** Can we accurately predict flight delays using the available features?  
   - **Objective:** Develop and evaluate predictive models to assess which features (e.g., scheduled departure time, taxi times, distance) are most informative.

7. **Cancellation & Diversion Analysis:**  
   - **Q:** What factors most frequently lead to flight cancellations or diversions, and how do these events correlate with delay patterns?  
   - **Objective:** Examine the relationship between operational issues and cancellations/diversions to identify potential preventive measures.

8. **Operational Efficiency & Turnaround Times:**  
   - **Q:** How do taxi out and taxi in times affect overall flight duration and delays?  
   - **Objective:** Study the efficiency of ground operations and their contribution to the final delay metrics.

## Project Structure

The repository is organized as follows:

