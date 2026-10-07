"""
console/app.py
----------------------------------------------------------------
Analyst console (Streamlit, non-ML component) :8501.

Tabs: Simulate Payment (calls triage-api live), Case Queue (calls
case-service), Monitoring (calls monitor-service). This is the UI used for
the assignment's screenshots.
----------------------------------------------------------------
"""
import os

import httpx
import streamlit as st

TRIAGE_API_URL = os.environ.get("TRIAGE_API_URL", "http://localhost:8000")
CASE_SERVICE_URL = os.environ.get("CASE_SERVICE_URL", "http://localhost:8002")
MONITOR_URL = os.environ.get("MONITOR_URL", "http://localhost:8003")
API_KEY = "demo-key-g13"

st.set_page_config(page_title="Fraud Triage Console — Group 13", layout="wide")
st.title("💳 Real-Time Payment Fraud Risk Triage — Group 13")

PRESETS = {
    "Genuine payment": dict(txn_id="DEMO-GENUINE", step=120, type="TRANSFER", amount=500.0,
                             nameOrig="C-PAYER-1", oldbalanceOrg=5000.0,
                             nameDest="C-PAYEE-1", oldbalanceDest=3000.0),
    "Account takeover (drains balance)": dict(
        txn_id="DEMO-ATO", step=50, type="TRANSFER", amount=9500.0,
        nameOrig="C-PAYER-2", oldbalanceOrg=9500.0, nameDest="C-NEW-PAYEE", oldbalanceDest=0.0,
    ),
    "Mule fan-in (cash-out)": dict(
        txn_id="DEMO-MULE", step=200, type="CASH_OUT", amount=45000.0,
        nameOrig="C-PAYER-3", oldbalanceOrg=45000.0, nameDest="C-MULE-1", oldbalanceDest=0.0,
    ),
}

tab_sim, tab_cases, tab_monitor = st.tabs(["🧪 Simulate Payment", "📋 Case Queue", "📈 Monitoring"])

with tab_sim:
    preset_name = st.selectbox("Preset scenario", list(PRESETS.keys()))
    preset = PRESETS[preset_name]
    col1, col2 = st.columns(2)
    with col1:
        txn_id = st.text_input("Transaction ID", preset["txn_id"])
        ptype = st.selectbox("Type", ["TRANSFER", "CASH_OUT"],
                              index=["TRANSFER", "CASH_OUT"].index(preset["type"]))
        amount = st.number_input("Amount (₹)", value=preset["amount"], min_value=0.01)
        step = st.number_input("Step (hour index)", value=preset["step"], min_value=0)
    with col2:
        name_orig = st.text_input("Sender account", preset["nameOrig"])
        oldbalance_orig = st.number_input("Sender balance before", value=preset["oldbalanceOrg"])
        name_dest = st.text_input("Receiver account", preset["nameDest"])
        oldbalance_dest = st.number_input("Receiver balance before", value=preset["oldbalanceDest"])

    if st.button("Submit payment", type="primary"):
        payload = {
            "txn_id": txn_id, "step": int(step), "type": ptype, "amount": amount,
            "nameOrig": name_orig, "oldbalanceOrg": oldbalance_orig,
            "nameDest": name_dest, "oldbalanceDest": oldbalance_dest,
        }
        try:
            resp = httpx.post(
                f"{TRIAGE_API_URL}/v1/payments/triage", json=payload,
                headers={"X-API-Key": API_KEY}, timeout=5.0,
            )
            resp.raise_for_status()
            result = resp.json()
            color = {"ALLOW": "green", "STEP_UP": "orange", "BLOCK": "red"}[result["decision"]]
            st.markdown(f"### Decision: :{color}[{result['decision']}]")
            st.metric("Risk score", f"{result['risk_score']:.4f}")
            st.metric("Latency", f"{result['latency_ms']:.1f} ms")
            if result["degraded"]:
                st.warning("Scoring service degraded — rules-only fallback used")
            st.write("**Reason codes:**")
            for r in result["reason_codes"]:
                st.write(f"- {r}")
            with st.expander("Raw response"):
                st.json(result)
        except Exception as exc:
            st.error(f"Request failed: {exc}")

with tab_cases:
    status_filter = st.radio("Show", ["OPEN", "RESOLVED", "All"], horizontal=True)
    try:
        params = {} if status_filter == "All" else {"status": status_filter}
        cases = httpx.get(f"{CASE_SERVICE_URL}/cases", params=params, timeout=5.0).json()
        if not cases:
            st.info("No cases yet. Submit a BLOCK or high-value STEP_UP payment above.")
        for case in cases:
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 1, 1])
                c1.write(f"**{case['case_id']}** · txn `{case['txn_id']}` · {case['decision']}")
                c1.caption(" | ".join(case["reason_codes"]))
                c2.metric("Priority", f"{case['priority']:,.0f}")
                c3.metric("Amount", f"₹{case['amount']:,.0f}")
                if case["status"] == "OPEN":
                    b1, b2 = st.columns(2)
                    if b1.button("Confirm fraud", key=f"fraud-{case['case_id']}"):
                        httpx.post(f"{CASE_SERVICE_URL}/cases/{case['case_id']}/resolve",
                                   json={"verdict": "CONFIRMED_FRAUD"})
                        st.rerun()
                    if b2.button("Mark genuine", key=f"genuine-{case['case_id']}"):
                        httpx.post(f"{CASE_SERVICE_URL}/cases/{case['case_id']}/resolve",
                                   json={"verdict": "GENUINE"})
                        st.rerun()
                else:
                    st.caption(f"Resolved: {case['verdict']}")
    except Exception as exc:
        st.error(f"Could not reach case-service: {exc}")

with tab_monitor:
    try:
        m = httpx.get(f"{MONITOR_URL}/metrics", timeout=5.0).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total decisions", m["total_decisions"])
        c2.metric("PSI drift", f"{m['psi_drift']:.3f}" if m["psi_drift"] is not None else "n/a")
        c3.metric("Drift alert", "🔴 YES" if m["drift_alert"] else "🟢 no")
        st.write("**Decision mix:**")
        st.bar_chart(m["decision_mix"])
        if st.button("Refresh"):
            st.rerun()
    except Exception as exc:
        st.error(f"Could not reach monitor-service: {exc}")
