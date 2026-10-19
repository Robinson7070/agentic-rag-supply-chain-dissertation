"""
Supply Chain Document Intelligence Dashboard
Agentic RAG System for Automated Document Intelligence in Supply Chain Operations
Victor Chukwudi Robinson — MSc AI & Data Science, University of East London
Dissertation 2026
"""

import streamlit as st
import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'src')
from metrics_utils import binary_metrics

st.set_page_config(
    page_title="Supply Chain Document Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1F3864, #2E6DB4);
    padding: 20px 30px; border-radius: 10px;
    color: white; margin-bottom: 20px;
}
.metric-card {
    background: #f8f9fa; border-left: 4px solid #2E6DB4;
    padding: 15px; border-radius: 5px; margin: 5px 0;
}
.anomaly-card {
    background: #fff5f5; border-left: 4px solid #e53e3e;
    padding: 15px; border-radius: 5px; margin: 10px 0;
}
.consistent-card {
    background: #f0fff4; border-left: 4px solid #38a169;
    padding: 15px; border-radius: 5px; margin: 10px 0;
}
.thought-box {
    background: #ebf8ff; border-left: 3px solid #3182ce;
    padding: 10px 15px; border-radius: 4px; margin: 5px 0; font-size: 0.9em;
}
.action-box {
    background: #faf5ff; border-left: 3px solid #805ad5;
    padding: 10px 15px; border-radius: 4px; margin: 5px 0; font-size: 0.9em;
}
.observation-box {
    background: #f0fff4; border-left: 3px solid #38a169;
    padding: 10px 15px; border-radius: 4px; margin: 5px 0; font-size: 0.9em;
}
.hitl-pending {
    background: #fffbeb; border: 2px solid #d69e2e;
    padding: 15px; border-radius: 8px; margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:1.8em;">🔍 Supply Chain Document Intelligence</h1>
    <p style="margin:5px 0 0 0; opacity:0.9;">Agentic RAG System for Automated Anomaly Detection</p>
    <p style="margin:2px 0 0 0; opacity:0.7; font-size:0.85em;">
        Victor Chukwudi Robinson | MSc AI & Data Science | University of East London | 2026
    </p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    try:
        from agent import get_resources
        index, metadata, document_results = get_resources()
        return index, metadata, document_results, None
    except Exception as e:
        return None, None, None, str(e)


def _first_existing(*candidates):
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    return None


@st.cache_data
def load_evaluation_results():
    # Committed 20-set results first; fall back to a locally generated vector_store/ copy.
    results_path = _first_existing("results/evaluation_results_20sets.json",
                                   "vector_store/evaluation_results.json")
    if results_path:
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


@st.cache_data
def load_anomaly_log():
    log_path = _first_existing("results/anomaly_log.json", "vector_store/anomaly_log.json")
    if log_path:
        with open(log_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


with st.sidebar:
    st.markdown("### System Configuration")
    index, metadata, document_results, load_error = load_system()

    if load_error:
        st.error(f"System not loaded: {load_error}")
        st.info("Run python src/embeddings_store.py first")
        st.stop()
    else:
        st.success("Vector store loaded")
        st.metric("Documents indexed", len(document_results) if document_results else 0)
        st.metric("Total chunks", len(metadata) if metadata else 0)

    st.markdown("---")

    if metadata:
        po_refs = sorted(set(
            m['po_ref'] for m in metadata
            if m['po_ref'] not in ['NOT_FOUND', 'NOT_PROVIDED']
            and not m['po_ref'].startswith('PO-(informal')
        ))
    else:
        po_refs = []

    st.markdown("### Select Document Set")
    selected_po = st.selectbox(
        "Purchase Order Reference", po_refs,
        index=po_refs.index('PO-2026-2001') if 'PO-2026-2001' in po_refs else 0
    )

    st.markdown("---")
    st.markdown("### Agent Mode")
    agent_mode = st.radio(
        "Verification method",
        ["Structured Pipeline", "ReAct Agent"],
        help="Structured Pipeline is deterministic. ReAct Agent uses GPT-4o reasoning."
    )

    agent_query = ""
    if agent_mode == "ReAct Agent":
        agent_query = st.text_area(
            "Ask anything (optional)",
            height=90,
            placeholder=(
                "Leave blank to run standard verification, or ask:\n"
                "Which supplier has the highest average unit price?\n"
                "Compare PO-2026-2001 and PO-2026-2012\n"
                "Are there any documents missing a delivery note?"
            ),
        )

    st.markdown("---")
    st.markdown("""
    **Architecture:**
    - LangChain ReAct Agent (GPT-4o)
    - FAISS Vector Store (1536-dim)
    - Structure-aware chunking
    - Dual HITL intervention
    """)


tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Dashboard",
    "Verify Document Set",
    "Evaluation Dashboard",
    "RAG Comparison",
    "Anomaly Report",
    "System Architecture"
])


# TAB 0 — EXECUTIVE DASHBOARD
with tab0:
    st.markdown("### Executive Summary")
    st.markdown("Overview of the dissertation evaluation run (20 synthetic document sets)")

    eval_results = load_evaluation_results()
    anomaly_log = load_anomaly_log()

    # Calculate all metrics
    total_pos = len(eval_results) if eval_results else 0
    anomaly_sets = [r for r in eval_results if not r['consistent']] if eval_results else []
    consistent_sets = [r for r in eval_results if r['consistent']] if eval_results else []
    all_anomalies = anomaly_log if anomaly_log else []
    pending_review = [a for a in all_anomalies if a.get('status') == 'pending_human_review']
    high_risk = [a for a in all_anomalies if a.get('severity') == 'high']
    confirmed = [a for a in all_anomalies if a.get('status') == 'confirmed_by_reviewer']

    # Financial impact: one figure per (PO, document pair). A single price error is logged
    # both as a unit-price mismatch and as a total mismatch (the total includes VAT), so only
    # the largest amount per document pair is counted, to avoid double counting.
    import re
    impact_by_pair = {}
    for anomaly in all_anomalies:
        amount = re.search(r'£([\d,]+\.?\d*)', anomaly.get('impact', ''))
        if amount:
            try:
                value = float(amount.group(1).replace(',', ''))
            except ValueError:
                continue
            key = (anomaly.get('po_ref'), anomaly.get('document1'), anomaly.get('document2'))
            impact_by_pair[key] = max(impact_by_pair.get(key, 0.0), value)
    financial_impact = sum(impact_by_pair.values())

    # Set-level metrics, computed from the results file (see src/metrics_utils.py)
    set_metrics = binary_metrics(eval_results) if eval_results else None

    st.markdown("---")

    # ROW 1 — Key operational metrics
    st.markdown("#### Operational Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Total POs Processed",
            total_pos,
            help="Total purchase order document sets verified"
        )
    with c2:
        flagged_share = f"{len(anomaly_sets)/total_pos*100:.0f}%" if total_pos > 0 else "0%"
        st.metric(
            "Anomaly Sets Flagged",
            len(anomaly_sets),
            delta=f"{flagged_share} of sets flagged",
            delta_color="off",
            help="Document sets the system flagged as containing at least one anomaly"
        )
    with c3:
        st.metric(
            "High-Risk Invoices",
            len(high_risk),
            delta="Require urgent review",
            delta_color="inverse",
            help="Anomalies flagged as high severity"
        )
    with c4:
        st.metric(
            "Pending Human Reviews",
            len(pending_review),
            delta="Awaiting confirmation",
            delta_color="inverse" if len(pending_review) > 0 else "normal",
            help="Flagged anomalies awaiting human confirmation"
        )

    st.markdown("---")

    # ROW 2 — Financial and performance metrics
    st.markdown("#### Financial & Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Estimated Financial Impact",
            f"£{financial_impact:,.2f}",
            help="Largest discrepancy per document pair, summed across the log (price and total mismatches on the same invoice are not double counted)"
        )
    with c2:
        st.metric(
            "Consistent Sets",
            len(consistent_sets),
            delta="no anomalies flagged",
            delta_color="off",
            help="Document sets the system judged internally consistent"
        )
    with c3:
        if set_metrics:
            st.metric(
                "Set-level Precision",
                f"{set_metrics['precision']:.0%}",
                delta=f"{set_metrics['fp']} false positives",
                delta_color="off",
                help="Of the sets flagged as anomalous, the share that really were (computed from the results file)"
            )
        else:
            st.metric("Set-level Precision", "n/a", help="No results file found")
    with c4:
        if set_metrics:
            st.metric(
                "Set-level Recall",
                f"{set_metrics['recall']:.0%}",
                delta=f"F1 Score: {set_metrics['f1']:.3f}",
                delta_color="off",
                help="Of the sets that really contain anomalies, the share the system flagged (computed from the results file)"
            )
        else:
            st.metric("Set-level Recall", "n/a", help="No results file found")

    st.markdown("---")

    # ROW 3 — Anomaly breakdown and status
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Anomaly Breakdown by Type")
        type_counts = {}
        for a in all_anomalies:
            t = a.get('anomaly_type', 'unknown').replace('_', ' ').title()
            type_counts[t] = type_counts.get(t, 0) + 1

        if type_counts:
            for atype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                pct = count / len(all_anomalies) * 100 if all_anomalies else 0
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{atype}</strong><br>
                    {count} detected ({pct:.0f}%)
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No anomalies logged yet")

    with col2:
        st.markdown("#### Review Status")
        status_counts = {
            'Pending Review': len(pending_review),
            'Confirmed': len(confirmed),
            'Rejected': len([a for a in all_anomalies if a.get('status') == 'rejected_false_positive'])
        }
        colours = {'Pending Review': '#d69e2e', 'Confirmed': '#38a169', 'Rejected': '#718096'}
        for status, count in status_counts.items():
            colour = colours.get(status, '#666')
            st.markdown(f"""
            <div style="background:#f8f9fa; border-left:4px solid {colour};
                        padding:15px; border-radius:5px; margin:5px 0;">
                <strong>{status}:</strong> {count} anomalies
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Severity Distribution")
        sev_counts = {}
        for a in all_anomalies:
            s = a.get('severity', 'unknown').title()
            sev_counts[s] = sev_counts.get(s, 0) + 1

        sev_colours = {'High': '#e53e3e', 'Medium': '#d69e2e', 'Low': '#38a169'}
        for sev, count in sev_counts.items():
            colour = sev_colours.get(sev, '#666')
            st.markdown(f"""
            <div style="background:#f8f9fa; border-left:4px solid {colour};
                        padding:10px 15px; border-radius:5px; margin:5px 0;">
                <strong style="color:{colour}">{sev} Severity:</strong> {count}
            </div>
            """, unsafe_allow_html=True)

    with col3:
        st.markdown("#### Recommendations")

        recommendations = []

        if len(pending_review) > 0:
            recommendations.append({
                'priority': 'URGENT',
                'colour': '#e53e3e',
                'text': f'Review {len(pending_review)} pending anomaly/anomalies immediately before approving payment'
            })

        if len(high_risk) > 0:
            recommendations.append({
                'priority': 'HIGH',
                'colour': '#e53e3e',
                'text': f'{len(high_risk)} high-severity anomaly/anomalies detected — escalate to procurement manager'
            })

        price_mismatches = [a for a in all_anomalies if a.get('anomaly_type') == 'price_mismatch']
        if price_mismatches:
            recommendations.append({
                'priority': 'HIGH',
                'colour': '#d69e2e',
                'text': f'{len(price_mismatches)} price mismatch(es) detected — contact suppliers to reconcile invoices'
            })

        qty_mismatches = [a for a in all_anomalies if a.get('anomaly_type') == 'quantity_mismatch']
        if qty_mismatches:
            recommendations.append({
                'priority': 'MEDIUM',
                'colour': '#d69e2e',
                'text': f'{len(qty_mismatches)} quantity discrepancy/discrepancies — verify delivery receipts and update inventory'
            })

        missing_docs = [a for a in all_anomalies if a.get('anomaly_type') == 'missing_document']
        if missing_docs:
            recommendations.append({
                'priority': 'HIGH',
                'colour': '#e53e3e',
                'text': f'{len(missing_docs)} missing document(s) — do not process payment without complete documentation'
            })

        date_issues = [a for a in all_anomalies if a.get('anomaly_type') == 'date_inconsistency']
        if date_issues:
            recommendations.append({
                'priority': 'MEDIUM',
                'colour': '#d69e2e',
                'text': f'{len(date_issues)} date inconsistency/inconsistencies — verify document authenticity with suppliers'
            })

        if financial_impact > 0:
            recommendations.append({
                'priority': 'INFO',
                'colour': '#2E6DB4',
                'text': f'Total financial exposure: £{financial_impact:,.2f} — recover overcharges before payment processing'
            })

        if not recommendations:
            st.markdown("""
            <div class="consistent-card">
                <strong>All Clear</strong><br>
                No outstanding actions required. All verified document sets are consistent.
            </div>
            """, unsafe_allow_html=True)
        else:
            for rec in recommendations:
                st.markdown(f"""
                <div style="background:#fff; border-left:4px solid {rec['colour']};
                            padding:12px 15px; border-radius:5px; margin:8px 0;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <span style="background:{rec['colour']}; color:white; padding:2px 8px;
                                 border-radius:3px; font-size:0.75em; font-weight:bold;">
                        {rec['priority']}
                    </span>
                    <p style="margin:8px 0 0 0; font-size:0.9em;">{rec['text']}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ROW 4 — Recent anomalies table
    st.markdown("#### Recent Anomalies")
    if all_anomalies:
        recent = sorted(all_anomalies, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]

        header_cols = st.columns([1, 2, 2, 2, 2, 1])
        headers = ['ID', 'PO Reference', 'Type', 'Impact', 'Severity', 'Status']
        for col, h in zip(header_cols, headers):
            col.markdown(f"**{h}**")

        st.markdown("<hr style='margin:5px 0'>", unsafe_allow_html=True)

        for a in recent:
            row_cols = st.columns([1, 2, 2, 2, 2, 1])
            severity = a.get('severity', '')
            sev_colour = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
            status_icon = {'pending_human_review': '⏳', 'confirmed_by_reviewer': '✅', 'rejected_false_positive': '❌'}.get(a.get('status', ''), '❓')

            row_cols[0].markdown(f"`{a.get('anomaly_id','N/A')}`")
            row_cols[1].markdown(a.get('po_ref', 'N/A'))
            row_cols[2].markdown(a.get('anomaly_type', 'N/A').replace('_', ' ').title())
            impact = a.get('impact', 'N/A')
            row_cols[3].markdown(impact[:40] + '...' if len(impact) > 40 else impact)
            row_cols[4].markdown(f"{sev_colour} {severity.title()}")
            row_cols[5].markdown(status_icon)
    else:
        st.info("No anomalies logged yet. Run a verification to populate this table.")




# TAB 1
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Document Set")
        st.info(f"**Selected PO:** {selected_po}")

        if metadata:
            set_docs = {}
            for m in metadata:
                if m['po_ref'] == selected_po:
                    dt = m['document_type']
                    if dt not in set_docs:
                        set_docs[dt] = m['filename']

            st.markdown("**Documents found:**")
            for dt in ['purchase_order', 'invoice', 'delivery_note']:
                label = dt.replace('_', ' ').title()
                if dt in set_docs:
                    st.markdown(f"✅ **{label}**  \n`{set_docs[dt]}`")
                else:
                    st.markdown(f"❌ **{label}** — *missing*")

        st.markdown("---")
        verify_btn = st.button("Run Verification", type="primary", use_container_width=True)

    with col2:
        st.markdown("### Verification Results")

        if verify_btn:
            if agent_mode == "Structured Pipeline":
                with st.spinner("Running structured verification pipeline..."):
                    try:
                        import agent as agent_module
                        agent_module._anomaly_log = []
                        from verification_pipeline import verify_document_set
                        result = verify_document_set(selected_po, verbose=False)

                        if result['consistent']:
                            st.markdown('<div class="consistent-card"><h3 style="color:#38a169;margin:0">CONSISTENT — No anomalies detected</h3></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="anomaly-card"><h3 style="color:#e53e3e;margin:0">ANOMALY DETECTED — {len(result["anomalies"])} issue(s) found</h3></div>', unsafe_allow_html=True)
                            for anomaly in result['anomalies']:
                                atype = anomaly['type'].replace('_', ' ').title()
                                details = anomaly.get('details', '')
                                st.markdown(f"**{atype}:** {details}")

                        st.markdown("---")
                        st.markdown("**Documents verified:**")
                        c1, c2, c3 = st.columns(3)
                        for col, dt in zip([c1, c2, c3], ['purchase_order', 'invoice', 'delivery_note']):
                            with col:
                                label = dt.replace('_', ' ').title()
                                if dt in result['documents_checked']:
                                    st.success(label)
                                else:
                                    st.error(f"Missing: {label}")

                        if result.get('hitl_triggered'):
                            st.warning("HITL intervention triggered — low confidence extraction")
                        
                        # Processing Statistics
                        stats = result.get('stats', {})
                        if stats:
                            st.markdown("---")
                            st.markdown("**Processing Statistics:**")
                            s1, s2, s3, s4 = st.columns(4)
                            s1.metric("Documents Processed", stats.get('documents_processed', 0))
                            s2.metric("Tools Available", stats.get('tools_used', 5))
                            s3.metric("Execution Time", f"{stats.get('execution_time_seconds', 0):.2f}s")
                            s4.metric("Anomalies Found", len(result.get('anomalies', [])))

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            else:
                with st.spinner("Running ReAct agent... (30-60 seconds)"):
                    try:
                        import agent as agent_module
                        agent_module._anomaly_log = []
                        from agent import create_agent
                        agent_exec = create_agent()

                        # Open-ended query if the user typed one, else default verification
                        custom = agent_query.strip()
                        if custom:
                            query = custom
                        else:
                            query = (
                                f"Please verify the supply chain documents for purchase order {selected_po}. "
                                f"Check unit prices, quantities, totals and flag any discrepancies."
                            )

                        result = agent_exec.invoke({"input": query})
                        output = result.get('output', 'No output')
                        steps = result.get('intermediate_steps', [])

                        if 'ANOMALY' in output.upper():
                            st.markdown('<div class="anomaly-card"><h3 style="color:#e53e3e;margin:0">ANOMALY DETECTED</h3></div>', unsafe_allow_html=True)
                        elif 'CONSISTENT' in output.upper():
                            st.markdown('<div class="consistent-card"><h3 style="color:#38a169;margin:0">CONSISTENT</h3></div>', unsafe_allow_html=True)

                        st.markdown("**Agent Final Report:**")
                        st.markdown(output)

                        if steps:
                            st.markdown("---")
                            st.markdown("**Agent Reasoning Trace (ReAct Loop):**")
                            for i, (action, observation) in enumerate(steps):
                                tool_name = getattr(action, 'tool', 'unknown')
                                tool_input = getattr(action, 'tool_input', '')
                                log = getattr(action, 'log', '')
                                thought = ''
                                if 'Thought:' in log:
                                    thought = log.split('Thought:')[-1].split('Action:')[0].strip()

                                with st.expander(f"Step {i+1}: {tool_name}", expanded=i < 3):
                                    if thought:
                                        st.markdown(f'<div class="thought-box"><strong>Thought:</strong> {thought}</div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="action-box"><strong>Action:</strong> <code>{tool_name}</code> | <strong>Input:</strong> {tool_input}</div>', unsafe_allow_html=True)
                                    obs_preview = str(observation)[:400] + "..." if len(str(observation)) > 400 else str(observation)
                                    st.markdown(f'<div class="observation-box"><strong>Observation:</strong> {obs_preview}</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Agent error: {str(e)}")

            # HITL
            st.markdown("---")
            st.markdown("### Human-in-the-Loop Review")
            anomaly_log = load_anomaly_log()
            pending = [a for a in anomaly_log if a.get('status') == 'pending_human_review']

            if pending:
                st.warning(f"{len(pending)} anomaly/anomalies pending human review")
                for anomaly in pending:
                    severity = anomaly.get('severity', 'unknown')
                    severity_color = {'high': '#e53e3e', 'medium': '#d69e2e'}.get(severity, '#666')
                    st.markdown(f"""
                    <div class="hitl-pending">
                        <strong>{anomaly.get('anomaly_id','N/A')}</strong> | 
                        {anomaly.get('anomaly_type','N/A').replace('_',' ').title()} | 
                        <span style="color:{severity_color}"><strong>{severity.upper()}</strong></span><br>
                        <strong>Impact:</strong> {anomaly.get('impact','N/A')}<br>
                        <strong>Documents:</strong> {anomaly.get('document1','N/A')} vs {anomaly.get('document2','N/A')}
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(f"Confirm Anomaly", key=f"confirm_{anomaly.get('anomaly_id')}"):
                            anomaly['status'] = 'confirmed_by_reviewer'
                            anomaly['reviewed_at'] = datetime.now().isoformat()
                            with open("vector_store/anomaly_log.json", 'w') as f:
                                json.dump(anomaly_log, f, indent=2)
                            st.success("Confirmed")
                            st.cache_data.clear()
                            st.rerun()
                    with c2:
                        if st.button(f"Reject (False Positive)", key=f"reject_{anomaly.get('anomaly_id')}"):
                            anomaly['status'] = 'rejected_false_positive'
                            anomaly['reviewed_at'] = datetime.now().isoformat()
                            with open("vector_store/anomaly_log.json", 'w') as f:
                                json.dump(anomaly_log, f, indent=2)
                            st.info("Rejected")
                            st.cache_data.clear()
                            st.rerun()
            else:
                st.success("No anomalies pending review")


# TAB 2
with tab2:
    st.markdown("### Full System Evaluation — 20 Document Sets")
    eval_results = load_evaluation_results()

    if not eval_results:
        st.info("No evaluation results found.")
        if st.button("Run Full Evaluation"):
            with st.spinner("Running full evaluation..."):
                try:
                    import agent as agent_module
                    agent_module._anomaly_log = []
                    from verification_pipeline import run_full_evaluation
                    eval_results = run_full_evaluation(verbose=False)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        total = len(eval_results)
        anomaly_sets = [r for r in eval_results if not r['consistent']]
        consistent_sets = [r for r in eval_results if r['consistent']]
        total_anomalies = sum(len(r['anomalies']) for r in eval_results)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sets Evaluated", total)
        c2.metric("Anomaly Sets", len(anomaly_sets))
        c3.metric("Consistent Sets", len(consistent_sets))
        c4.metric("Total Anomalies", total_anomalies)
        c5.metric("Set-level F1", f"{binary_metrics(eval_results)['f1']:.3f}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Performance Metrics")
            m = binary_metrics(eval_results)
            st.caption("Set-level results (anomalous vs consistent) on 20 synthetic sets, computed from the results "
                       "file. Anomaly-type identification metrics are reported in the dissertation and are not "
                       "recomputed here.")
            for metric, value in {
                "Precision": f"{m['precision']:.0%}", "Recall": f"{m['recall']:.0%}", "F1 Score": f"{m['f1']:.3f}",
                "True Negatives": f"{m['tn']}/{m['tn'] + m['fp']}", "True Positives": f"{m['tp']}/{m['tp'] + m['fn']}",
                "False Positives": str(m['fp'])
            }.items():
                st.markdown(f'<div class="metric-card"><strong>{metric}:</strong> {value}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### Results by Set")
            for result in eval_results:
                po = result['po_ref']
                if result['consistent']:
                    st.markdown(f"✅ `{po}` — Consistent")
                else:
                    types = list(set(a['type'].replace('_',' ') for a in result['anomalies']))
                    st.markdown(f"⚠️ `{po}` — {', '.join(types)}")

        st.markdown("---")
        st.markdown("### Anomaly Type Breakdown")
        all_types = {}
        for r in eval_results:
            for a in r['anomalies']:
                t = a['type']
                all_types[t] = all_types.get(t, 0) + 1

        if all_types:
            cols = st.columns(len(all_types))
            for i, (t, count) in enumerate(all_types.items()):
                cols[i].metric(t.replace('_',' ').title(), count)


# TAB 3

# TAB 3 — RAG COMPARISON
with tab3:
    st.markdown("### Traditional RAG vs Agentic RAG — Comparative Analysis")
    st.markdown("Empirical comparison demonstrating the value of the agentic approach")

    # Load comparison results
    comp_path = _first_existing("results/comparison_results.json", "vector_store/comparison_results.json")

    if not comp_path:
        st.info("No comparison results found.")
        if st.button("Run Traditional RAG Evaluation"):
            with st.spinner("Running traditional RAG baseline on 6 test cases... (takes ~30 seconds)"):
                try:
                    from traditional_rag import run_comparison_evaluation
                    run_comparison_evaluation()
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        with open(comp_path, encoding='utf-8') as f:
            comp_results = json.load(f)

        # Summary metrics comparison
        st.markdown("#### Performance Comparison")
        col1, col2 = st.columns(2)

        with col1:
            _n = len(comp_results)
            _trad_ok = sum(1 for r in comp_results if r['trad_detected'] == r['ground_truth_anomaly'])
            _trad_fp = sum(1 for r in comp_results if r['trad_detected'] and not r['ground_truth_anomaly'])
            _trad_acc = f"{_trad_ok / _n:.0%}" if _n else "n/a"
            _trad_time = sum(r.get('trad_time', 0) for r in comp_results) / _n if _n else 0
            _trad_tok = sum(r.get('trad_tokens', 0) for r in comp_results) / _n if _n else 0
            st.markdown(f"""
            <div style="background:#fff5f5; border:2px solid #e53e3e;
                        padding:20px; border-radius:10px; text-align:center;">
                <h3 style="color:#e53e3e; margin:0">Traditional RAG</h3>
                <p style="font-size:0.85em; color:#666; margin:5px 0">Lewis et al. (2020) baseline</p>
                <hr>
                <h2 style="color:#e53e3e; margin:10px 0">{_trad_acc}</h2>
                <p style="margin:0">Accuracy ({_trad_ok}/{_n} test cases)</p>
                <br>
                <p><strong>{_trad_fp}</strong> False Positive(s)</p>
                <p><strong>{_trad_time:.2f}s</strong> avg per query</p>
                <p><strong>{_trad_tok:.0f}</strong> avg tokens</p>
                <p>❌ No HITL</p>
                <p>❌ No evidence trail</p>
                <p>❌ No anomaly logging</p>
                <p>⚠️ Hallucination risk</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            _res = load_evaluation_results()
            _m = binary_metrics(_res) if _res else None
            # Like-for-like: the agentic verdicts on the SAME cases the baseline was run on
            _agentic_by_po = {r['po_ref']: (not r['consistent']) for r in (_res or [])}
            _h2h = [(r['ground_truth_anomaly'], _agentic_by_po.get(r['po_ref'])) for r in comp_results]
            _h2h_n = len(_h2h)
            _h2h_ok = sum(1 for gt, ag in _h2h if ag is not None and ag == gt)
            _h2h_acc = f"{_h2h_ok / _h2h_n:.0%}" if _h2h_n else "n/a"
            _acc20 = f"{_m['accuracy']:.0%} ({_m['tp'] + _m['tn']}/{_m['n']} sets)" if _m else "n/a"
            _f1 = f"{_m['f1']:.3f}" if _m else "n/a"
            _fp = _m['fp'] if _m else "n/a"
            _prec = f"{_m['precision']:.0%}" if _m else "n/a"
            st.markdown(f"""
            <div style="background:#f0fff4; border:2px solid #38a169;
                        padding:20px; border-radius:10px; text-align:center;">
                <h3 style="color:#38a169; margin:0">Agentic RAG</h3>
                <p style="font-size:0.85em; color:#666; margin:5px 0">This dissertation system</p>
                <hr>
                <h2 style="color:#38a169; margin:10px 0">{_h2h_acc}</h2>
                <p style="margin:0">Accuracy ({_h2h_ok}/{_h2h_n} test cases, the same cases as the baseline)</p>
                <br>
                <p><strong>Full 20-set evaluation:</strong> {_acc20}, F1 {_f1}</p>
                <p><strong>{_fp}</strong> False Positives — {_prec} Precision</p>
                <p><strong>Higher</strong> tokens (multi-step)</p>
                <p>✅ Dual HITL intervention</p>
                <p>✅ Full evidence traceability</p>
                <p>✅ Structured anomaly logging</p>
                <p>✅ Answers cite retrieved evidence</p>
                <p style="font-size:0.8em; color:#666; margin-top:12px">Six test cases are too few to support a significance claim; treat this comparison as indicative.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Detailed results table
        st.markdown("#### Case-by-Case Comparison")

        header_cols = st.columns([2, 2, 1, 1, 1])
        for col, h in zip(header_cols, ["PO Reference", "Ground Truth", "Traditional RAG", "Agentic RAG", "Difference"]):
            col.markdown(f"**{h}**")

        st.markdown("<hr style='margin:5px 0'>", unsafe_allow_html=True)

        # Agentic verdicts come from the evaluation results file (anomaly flagged = not consistent)
        agentic_results = {r['po_ref']: (not r['consistent']) for r in load_evaluation_results()}

        for r in comp_results:
            po = r["po_ref"]
            gt = r["ground_truth_anomaly"]
            trad = r["trad_detected"]
            agentic = agentic_results.get(po, None)

            trad_correct = (trad == gt)
            agentic_correct = (agentic == gt) if agentic is not None else None

            row = st.columns([2, 2, 1, 1, 1])
            row[0].markdown(f"`{po}`")
            row[1].markdown(f"{'⚠️ ANOMALY' if gt else '✅ CONSISTENT'}")
            row[2].markdown(f"{'✅' if trad_correct else '❌'} {'ANOMALY' if trad else 'CONSIST'}")
            row[3].markdown(f"{'✅' if agentic_correct else '❌'} {'ANOMALY' if agentic else 'CONSIST'}" if agentic is not None else "N/A")
            
            if agentic_correct and not trad_correct:
                row[4].markdown("🟢 Agentic wins")
            elif trad_correct and not agentic_correct:
                row[4].markdown("🔴 Trad wins")
            elif trad_correct and agentic_correct:
                row[4].markdown("⚪ Both correct")
            else:
                row[4].markdown("🟡 Both wrong")

        st.markdown("---")

        # Key findings
        st.markdown("#### Key Findings")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Where Traditional RAG fails:**

            ❌ **Missing document detection**
            Traditional RAG returned CONSISTENT for PO-2026-2007
            (missing delivery note). It cannot check document existence
            — it only retrieves similar text.

            ❌ **False positive on clean set**
            Traditional RAG incorrectly flagged PO-2026-1002 as
            anomalous. The agentic system correctly identified it
            as consistent.

            ❌ **Hallucination**
            Traditional RAG reported "$15.00 vs $14.50" for
            PO-2026-2002 — using dollar signs for a UK document.
            Values were confabulated, not retrieved.
            """)

        with col2:
            st.markdown("""
            **Why the agentic approach is superior:**

            ✅ **Systematic document verification**
            check_document_exists tool explicitly verifies all
            three documents are present before comparison begins.

            ✅ **Grounded comparison**
            compare_values tool compares the extracted numbers
            programmatically, rather than leaving the comparison to the model.

            ✅ **Precision is measured, not assumed**
            Set-level precision is computed from the results file;
            see the Evaluation Dashboard tab.

            ✅ **Complete audit trail**
            Every anomaly logged with source file, section,
            confidence score, and HITL status.
            """)

        st.markdown("---")
        st.markdown("""
        **Citation:** Lewis, P. et al. (2020). 'Retrieval-Augmented Generation for 
        Knowledge-Intensive NLP Tasks'. *Advances in Neural Information Processing Systems*, 33.
        """)

with tab4:
    st.markdown("### Anomaly Report")
    anomaly_log = load_anomaly_log()

    if not anomaly_log:
        st.info("No anomalies logged yet.")
    else:
        pending = [a for a in anomaly_log if a.get('status') == 'pending_human_review']
        confirmed = [a for a in anomaly_log if a.get('status') == 'confirmed_by_reviewer']
        rejected = [a for a in anomaly_log if a.get('status') == 'rejected_false_positive']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Flagged", len(anomaly_log))
        c2.metric("Pending Review", len(pending))
        c3.metric("Confirmed", len(confirmed))
        c4.metric("Rejected", len(rejected))

        st.markdown("---")
        status_filter = st.selectbox("Filter by status", ["All", "Pending Review", "Confirmed", "Rejected"])
        status_map = {"All": None, "Pending Review": "pending_human_review", "Confirmed": "confirmed_by_reviewer", "Rejected": "rejected_false_positive"}
        filtered = [a for a in anomaly_log if not status_map[status_filter] or a.get('status') == status_map[status_filter]]

        for anomaly in filtered:
            status_icon = {'pending_human_review': '⏳', 'confirmed_by_reviewer': '✅', 'rejected_false_positive': '❌'}.get(anomaly.get('status',''), '❓')
            severity = anomaly.get('severity', 'unknown')
            severity_color = {'high': '#e53e3e', 'medium': '#d69e2e'}.get(severity, '#666')

            with st.expander(f"{status_icon} {anomaly.get('anomaly_id','N/A')} | {anomaly.get('anomaly_type','N/A').replace('_',' ').title()} | PO: {anomaly.get('po_ref','N/A')}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Type:** {anomaly.get('anomaly_type','N/A').replace('_',' ').title()}")
                    st.markdown(f"**PO Reference:** {anomaly.get('po_ref','N/A')}")
                    st.markdown(f"**Field:** {anomaly.get('field','N/A')}")
                    st.markdown(f"**Values:** {anomaly.get('value1','N/A')} vs {anomaly.get('value2','N/A')}")
                with c2:
                    st.markdown(f"**Impact:** {anomaly.get('impact','N/A')}")
                    st.markdown(f"**Severity:** <span style='color:{severity_color}'>{severity.upper()}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Status:** {anomaly.get('status','N/A').replace('_',' ').title()}")
                    st.markdown(f"**Documents:** {anomaly.get('document1','N/A')} vs {anomaly.get('document2','N/A')}")
                
                # Evidence Traceability
                evidence = anomaly.get('evidence', {})
                if evidence:
                    st.markdown("---")
                    st.markdown("**Evidence Traceability:**")
                    ev_cols = st.columns(len(evidence))
                    for i, (doc_role, ev) in enumerate(evidence.items()):
                        with ev_cols[i]:
                            st.markdown(f"""
                            <div style="background:#f0f4ff; border:1px solid #c3d0f0;
                                        padding:10px; border-radius:6px; font-size:0.85em;">
                                <strong>{doc_role.upper()}</strong><br>
                                📄 {ev.get('file','N/A')}<br>
                                📑 Section: <code>{ev.get('section','N/A')}</code><br>
                                🎯 Confidence: {ev.get('confidence', 0):.0%}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Confidence breakdown
                conf_score = anomaly.get('confidence_score')
                checks_passed = anomaly.get('checks_passed', [])
                checks_failed = anomaly.get('checks_failed', [])
                
                if conf_score is not None or checks_passed or checks_failed:
                    st.markdown("---")
                    st.markdown("**Confidence Analysis:**")
                    if conf_score is not None:
                        st.progress(float(conf_score))
                        st.markdown(f"Overall Confidence: **{float(conf_score):.0%}**")
                    
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if checks_passed:
                            st.markdown("**Checks Passed:**")
                            for check in checks_passed:
                                st.markdown(f"✅ {check}")
                    with cc2:
                        if checks_failed:
                            st.markdown("**Checks Failed:**")
                            for check in checks_failed:
                                st.markdown(f"❌ {check}")


# TAB 5
with tab5:
    st.markdown("### System Architecture")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Seven-Layer Architecture

        **Layer 1 — Document Ingestion**
        - pdfplumber text extraction
        - Structure-aware chunking (header/line_item/summary)
        - 5 supplier format detection
        - Key field extraction with confidence scoring

        **Layer 2 — Embeddings & Vector Store**
        - OpenAI text-embedding-3-small (1536 dimensions)
        - Metadata enrichment before embedding
        - FAISS IndexFlatL2 exact search
        - 222 vectors across 56 documents

        **Layer 3 — Agent Orchestration**
        - LangChain ReAct framework (Yao et al., 2023)
        - GPT-4o for complex multi-step reasoning
        - 5 custom tools for supply chain verification

        **Layer 4 — Memory Layer**
        - ConversationBufferMemory for session context
        - Document result and anomaly log persistence

        **Layer 5 — Human-in-the-Loop**
        - Mid-retrieval: low confidence triggers HITL
        - Post-verification: anomalies pending review
        - Exception-handling design (Lazaros et al., 2026)

        **Layer 6 — Structured Verification Pipeline**
        - Deterministic orchestration of 5 tools
        - Date consistency checking
        - Missing document detection

        **Layer 7 — Dashboard Interface**
        - This Streamlit interface
        - Real-time verification and HITL buttons
        """)

    with col2:
        st.markdown("""
        #### Five Agent Tools

        **1. search_documents**
        Semantic search with metadata filtering across FAISS.

        **2. compare_values**
        Numerical comparison between document fields.
        Returns MATCH or DISCREPANCY with impact.

        **3. calculate_total**
        Calculates expected total from line items with VAT.

        **4. check_document_exists**
        Verifies document existence for a given PO reference.

        **5. flag_anomaly**
        Records anomalies for human review (HITL).

        ---

        #### Evaluation Results

        Computed from `results/evaluation_results_20sets.json` — see the Evaluation Dashboard tab.

        #### Contributions
        1. Structure-aware chunking for logistics documents
        2. Cross-document agentic reasoning across PO/Invoice/DN
        3. Open-source implementation for SMEs
        4. Controlled synthetic evaluation benchmark
        """)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#666;font-size:0.8em;'>Agentic RAG System for Supply Chain Document Intelligence | Victor Chukwudi Robinson | UEL 2026</p>", unsafe_allow_html=True)
