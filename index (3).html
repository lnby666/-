import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.graph_objects as go
import time
from datetime import datetime

# ==========================================
# 页面基础配置
# ==========================================
st.set_page_config(
    page_title="DSS Supply Chain Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 增强视觉硬核感
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 标题与侧边栏
# ==========================================
st.title("🛡️ 供应链“大脑-中枢-手脚”一体化应急响应系统")
st.caption("基于二次规划、BeFL 分布式协作与 JADC2 理念的系统 Demo")

with st.sidebar:
    st.header("🎮 控制台参数")
    st.markdown("---")
    lambda_val = st.slider("风险厌恶系数 (λ)", 0.1, 5.0, 1.5, help="系数越高，系统越倾向于规避风险而非追求低成本")
    scenario = st.selectbox("地缘政治场景", ["马六甲海峡出口管制", "东欧物流节点失效", "苏伊士运河突发封锁"])
    
    st.markdown("---")
    if st.button("🚀 启动全域应急响应", type="primary"):
        st.session_state['active'] = True
    if st.button("🔄 重置系统"):
        st.session_state['active'] = False

# ==========================================
# 核心逻辑：二次规划算法 (3.1.4)
# ==========================================
def solve_optimization(lmbda):
    n = 3
    # 预期效率 (国内路径, 现有路径, 应急冗余路径)
    returns = np.array([0.12, 0.18, 0.05])
    # 风险协方差矩阵 Sigma
    sigma = np.array([
        [0.05, 0.01, 0.00],
        [0.01, 0.30, 0.02], # 中间路径在冲突下波动巨大
        [0.00, 0.02, 0.02]
    ])
    
    w = cp.Variable(n)
    # 目标函数：最小化 (1/2 * w.T * Sigma * w - lambda * R.T * w)
    risk = 0.5 * cp.quad_form(w, sigma)
    profit = lmbda * (returns @ w)
    prob = cp.Problem(cp.Minimize(risk - profit), [cp.sum(w) == 1, w >= 0])
    prob.solve()
    return w.value

# ==========================================
# UI 布局
# ==========================================
col1, col2, col3 = st.columns([1, 1.2, 1])

# --- 3.3.1 风险预测大脑 ---
with col1:
    st.header("🧠 风险预测大脑")
    if st.session_state.get('active'):
        st.success("DSS 正在进行多源异构情报监测...")
        
        # 模拟沙盒演练 (3.2.1)
        with st.container():
            st.markdown("**EconomicSandbox 演练轨迹:**")
            messages = [
                "🕵️ [Agent-Intel]: 识别到地缘政治政策变动信号...",
                "🤝 [Agent-Trade]: 模拟对手方启动制裁博弈...",
                "📊 [Agent-Logistics]: 预测物理链路中断概率 > 82%..."
            ]
            for msg in messages:
                st.write(msg)
                time.sleep(0.4)
        
        st.metric("风险严重度评分", "8.9/10", delta="严重 (Critical)")
    else:
        st.info("请在左侧点击启动，激活系统感知层。")

# --- 3.3.2 方案决策中枢 ---
with col2:
    st.header("⚖️ 方案决策中枢")
    if st.session_state.get('active'):
        st.latex(r"\min \frac{1}{2} w^T \Sigma w - \lambda R^T w")
        
        # 运行算法
        weights = solve_optimization(lambda_val)
        
        # 权重展示图
        fig = go.Figure(data=[go.Pie(
            labels=['国内节点 (稳定)', '受损路径 (高波动)', '应急冗余 (备选)'],
            values=weights.flatten(),
            hole=.4,
            marker=dict(colors=['#00CC96', '#EF553B', '#636EFA'])
        )])
        fig.update_layout(title="DSS 动态资产分配权重", height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**CVaR 风险调整后成本评估完成**。建议向冗余路径倾斜资源。")
    else:
        st.write("等待大脑传递风险向量...")

# --- 3.3.3 敏捷执行手脚 ---
with col3:
    st.header("🦾 敏捷执行手脚")
    if st.session_state.get('active'):
        st.markdown("**全域应急 SOP (Reasoning-trace):**")
        st.checkbox("Step 1: 触发 BeFL 分布式协作指令", value=True)
        st.checkbox("Step 2: 动态重组物流路径 (海运 -> 多式联运)", value=True)
        st.checkbox("Step 3: 启动链上自动化履约结算", value=True)
        
        # 信任锚点展示 (3.2.2)
        st.markdown("---")
        st.markdown("**🔗 BeFL 信任锚点日志**")
        log_entry = {
            "Time": [datetime.now().strftime("%H:%M:%S")],
            "Hash": ["0x" + str(hash(time.time()))[:10] + "..."],
            "Status": ["Verified"]
        }
        st.table(pd.DataFrame(log_entry))
    else:
        st.write("等待决策中枢下达指令。")

# ==========================================
# 底部：商业价值展示面板
# ==========================================
st.markdown("---")
st.subheader("📈 系统效能对比 (实战演练表现)")
m1, m2, m3, m4 = st.columns(4)

if st.session_state.get('active'):
    m1.metric("潜在损失减少", "$4.2M", "82%")
    m2.metric("决策速度", "0.8s", "-95%")
    m3.metric("路径鲁棒性", "98.2%", "稳健")
    m4.metric("审计可追溯率", "100%", "BeFL加密")
else:
    m1.metric("潜在损失减少", "-", "-")
    m2.metric("决策速度", "-", "-")
    m3.metric("路径鲁棒性", "-", "-")
    m4.metric("审计可追溯率", "-", "-")
