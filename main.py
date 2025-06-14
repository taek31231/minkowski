import streamlit as st
import plotly.graph_objects as go
import numpy as np

# 상수 설정
C = 1.0  # 광속을 1로 설정하여 c*t 대신 t 사용 (단위 조절)

def lorentz_factor(v):
    """로렌츠 인자 감마를 계산합니다."""
    return 1 / np.sqrt(1 - (v**2 / C**2))

def lorentz_transform(t, x, v):
    """주어진 (t, x) 좌표를 속도 v로 움직이는 프라임 계 (t', x')로 변환합니다."""
    gamma = lorentz_factor(v)
    t_prime = gamma * (t - (v / C**2) * x)
    x_prime = gamma * (x - v * t)
    return t_prime, x_prime

def inverse_lorentz_transform(t_prime, x_prime, v):
    """주어진 (t', x') 좌표를 정지 계 (t, x)로 역변환합니다."""
    gamma = lorentz_factor(v)
    t = gamma * (t_prime + (v / C**2) * x_prime)
    x = gamma * (x_prime + v * t_prime)
    return t, x

def create_minkowski_diagram(v, show_length_contraction, show_time_dilation):
    gamma = lorentz_factor(v)

    # Plotly Figure 생성
    fig = go.Figure()

    # --- 1. 축 설정 ---
    axis_range = 5  # 축 범위
    fig.add_shape(type="line", x0=-axis_range, y0=0, x1=axis_range, y1=0,
                  line=dict(color="blue", width=2), name="x-axis (S)")
    fig.add_shape(type="line", x0=0, y0=-axis_range, x1=0, y1=axis_range,
                  line=dict(color="blue", width=2), name="t-axis (S)")

    # --- 2. 빛 원뿔 (Light Cone) ---
    # 빛의 세계선 (c=1 이므로 기울기 1 또는 -1)
    fig.add_shape(type="line", x0=-axis_range, y0=-axis_range, x1=axis_range, y1=axis_range,
                  line=dict(color="gray", width=1, dash="dash"), name="Light Cone (x=t)")
    fig.add_shape(type="line", x0=-axis_range, y0=axis_range, x1=axis_range, y1=-axis_range,
                  line=dict(color="gray", width=1, dash="dash"), name="Light Cone (x=-t)")

    # --- 3. 움직이는 관성계 (S')의 축 ---
    # t' 축 (S'에서 x'=0 인 점들의 궤적)
    # x = v*t 이므로 t = x/v. Plotly line은 (x0,y0,x1,y1) 이므로 t를 y로, x를 x로 생각
    # 기울기 1/v (y/x)
    if v != 0:
        t_prime_axis_x = np.array([-axis_range, axis_range])
        t_prime_axis_y = (1/v) * t_prime_axis_x if v != 0 else np.array([0,0]) # x=vt
        fig.add_shape(type="line", x0=t_prime_axis_x[0], y0=t_prime_axis_y[0],
                      x1=t_prime_axis_x[1], y1=t_prime_axis_y[1],
                      line=dict(color="red", width=2, dash="dot"), name="t'-axis (S')")

        # x' 축 (S'에서 t'=0 인 점들의 궤적)
        # t = v*x 이므로 y = v*x. 기울기 v (y/x)
        x_prime_axis_x = np.array([-axis_range, axis_range])
        x_prime_axis_y = v * x_prime_axis_x
        fig.add_shape(type="line", x0=x_prime_axis_x[0], y0=x_prime_axis_y[0],
                      x1=x_prime_axis_x[1], y1=x_prime_axis_y[1],
                      line=dict(color="red", width=2, dash="dot"), name="x'-axis (S')")

    # --- 4. 길이 수축 시각화 (Length Contraction) ---
    if show_length_contraction and v != 0:
        # S' 계에서 정지한 막대 (고유 길이 L0)
        L0 = 1.0 # S' 계에서의 고유 길이 (예: 1단위)

        # S' 계에서 막대의 양 끝점 (t'=0, x'=0)과 (t'=0, x'=L0)
        # 이를 S 계 좌표로 변환
        x_prime_start_lt = 0
        x_prime_end_lt = L0
        t_prime_lt = 0 # S' 계에서 동시에 측정

        t_s_start_lt, x_s_start_lt = inverse_lorentz_transform(t_prime_lt, x_prime_start_lt, v)
        t_s_end_lt, x_s_end_lt = inverse_lorentz_transform(t_prime_lt, x_prime_end_lt, v)

        # S 계에서 동시에 측정하기 위해 t_s_start_lt와 t_s_end_lt가 같아지도록 x축 평행선 사용
        # S 계에서 측정한 막대의 세계선 (시작점, 끝점)
        # (x', t') 기준으로 x'이 0인 세계선과 x'이 L0인 세계선 그리기
        
        # S' 계에서 x'=0 인 세계선
        t_prime_line = np.linspace(-axis_range, axis_range, 100)
        x_prime_0_line_t, x_prime_0_line_x = inverse_lorentz_transform(t_prime_line, 0, v)
        fig.add_trace(go.Scatter(x=x_prime_0_line_x, y=x_prime_0_line_t, mode='lines',
                                 line=dict(color='orange', dash='dash'), name='Worldline (x\'=0)'))
        
        # S' 계에서 x'=L0 인 세계선
        x_prime_L0_line_t, x_prime_L0_line_x = inverse_lorentz_transform(t_prime_line, L0, v)
        fig.add_trace(go.Scatter(x=x_prime_L0_line_x, y=x_prime_L0_line_t, mode='lines',
                                 line=dict(color='orange', dash='dash'), name=f'Worldline (x\'={L0})'))
        
        # S 계에서 동시에 측정하는 선 (t=0)
        measured_x_start = x_s_start_lt if v == 0 else x_prime_0_line_x[np.argmin(np.abs(x_prime_0_line_t - 0))]
        measured_x_end = x_s_end_lt if v == 0 else x_prime_L0_line_x[np.argmin(np.abs(x_prime_L0_line_t - 0))]
        
        # S 계에서 측정한 길이 표시
        fig.add_shape(type="line", x0=measured_x_start, y0=0, x1=measured_x_end, y1=0,
                      line=dict(color="green", width=3, dash="solid"), name="Measured Length in S")
        fig.add_annotation(x=(measured_x_start + measured_x_end)/2, y=0.2, text=f"L_S={measured_x_end - measured_x_start:.2f}", showarrow=False, font=dict(color="green"))
        fig.add_annotation(x=L0/2, y=v*(L0/2) + 0.2, text=f"L_S'={L0:.2f}", showarrow=False, font=dict(color="orange")) # S'에서의 길이 (대략적인 위치)

    # --- 5. 시간 팽창 시각화 (Time Dilation) ---
    if show_time_dilation and v != 0:
        # S' 계에서 고유 시간 간격 (델타 t_0)
        dt0 = 1.0 # S' 계에서의 고유 시간 (예: 1단위)

        # S' 계에서 한 시계의 두 사건 (x'=0, t'=0)과 (x'=0, t'=dt0)
        # 이를 S 계 좌표로 변환
        x_prime_td = 0 # S' 계에서 동일한 위치
        t_prime_start_td = 0
        t_prime_end_td = dt0

        t_s_start_td, x_s_start_td = inverse_lorentz_transform(t_prime_start_td, x_prime_td, v)
        t_s_end_td, x_s_end_td = inverse_lorentz_transform(t_prime_end_td, x_prime_td, v)

        # S' 계 시계의 세계선 (S' t-axis와 일치)
        # 이미 t' axis를 그렸으므로, 그 위에 점으로 표시하거나 더 진하게 표시
        fig.add_trace(go.Scatter(x=[x_s_start_td, x_s_end_td], y=[t_s_start_td, t_s_end_td], mode='lines+markers',
                                 marker=dict(color='purple', size=8), line=dict(color='purple', width=3, dash='solid'),
                                 name=f'Worldline of Clock (S\')'))
        fig.add_annotation(x=x_s_end_td+0.2, y=t_s_end_td+0.2, text=f"Δt_S={t_s_end_td - t_s_start_td:.2f}", showarrow=False, font=dict(color="purple"))
        fig.add_annotation(x=-0.2, y=dt0/2, text=f"Δt_S'={dt0:.2f}", showarrow=False, font=dict(color="red"))


    # --- 6. 레이아웃 설정 ---
    fig.update_layout(
        title=f'민코프스키 시공간 도표 (v/C = {v:.2f})',
        xaxis_title='x (공간)',
        yaxis_title='t (시간)',
        xaxis_range=[-axis_range, axis_range],
        yaxis_range=[-axis_range, axis_range],
        width=700,
        height=700,
        hovermode="closest",
        showlegend=True
    )
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')

    return fig

# --- Streamlit 앱 인터페이스 ---
st.set_page_config(layout="wide")

st.title("민코프스키 시공간 도표")

st.write("""
이 앱은 특수 상대성 이론의 민코프스키 시공간을 시각화합니다.
슬라이더를 조작하여 관성계의 상대 속도(v)를 변경해 보세요.
""")

# 사이드바에서 사용자 입력 받기
st.sidebar.header("설정")
v_speed = st.sidebar.slider("상대 속도 v/C (C=1)", min_value=-0.99, max_value=0.99, value=0.5, step=0.01)

st.sidebar.subheader("표시 옵션")
show_length_contraction = st.sidebar.checkbox("길이 수축 시각화", value=True)
show_time_dilation = st.sidebar.checkbox("시간 팽창 시각화", value=True)

# 도표 생성 및 표시
minkowski_fig = create_minkowski_diagram(v_speed, show_length_contraction, show_time_dilation)
st.plotly_chart(minkowski_fig, use_container_width=True)

st.subheader("설명")
st.markdown(f"""
* **파란색 축**: 정지한 관성계 ($S$)의 시간(t) 축과 공간(x) 축입니다.
* **빨간색 점선 축**: 속도 $v={v_speed:.2f}C$로 움직이는 관성계 ($S'$)의 시간(t') 축과 공간(x') 축입니다.
  * **t' 축**: $S'$ 계에서 $x'=0$인 점들의 세계선으로, $S$ 계에서는 $x=vt$ 직선에 해당합니다.
  * **x' 축**: $S'$ 계에서 $t'=0$인 점들의 세계선으로, $S$ 계에서는 $t=vx/C^2$ 직선에 해당합니다. (여기서는 $C=1$이므로 $t=vx$)
* **회색 점선**: 빛의 세계선입니다 ($x = \pm C t$). 빛은 어떤 관성계에서도 항상 $45^\circ$ 기울기를 가집니다.
""")

if show_length_contraction:
    st.markdown("""
    ---
    ### 길이 수축 (Length Contraction)
    * **주황색 점선**: $S'$ 계에 정지해 있는 막대의 양 끝의 세계선입니다. $S'$ 계에서는 이 막대의 길이가 $L_0 = 1.0$으로 측정됩니다.
    * **초록색 실선**: $S$ 계의 관찰자가 **동일한 시간(t=0)**에 막대의 양 끝을 측정한 길이입니다.
      * $S$ 계에서 측정한 길이 $L_S = L_0 / \gamma$ 입니다.
      * 현재 $v={v_speed:.2f}C$ 이므로 $\gamma = {lorentz_factor(v_speed):.2f}$ 이고, $L_S = {1.0 / lorentz_factor(v_speed):.2f}$ 입니다.
      * 도표에서 초록색 선이 주황색 점선(S' 계의 고유 길이)보다 짧아진 것을 확인할 수 있습니다.
    """)

if show_time_dilation:
    st.markdown("""
    ---
    ### 시간 팽창 (Time Dilation)
    * **보라색 실선**: $S'$ 계에 정지해 있는 시계의 세계선입니다 (t' 축 위에 있음). 이 시계가 1 단위 시간 ($\Delta t_0 = 1.0$) 동안 진행하는 두 사건을 나타냅니다.
    * **$S'$ 계의 고유 시간**: 시계가 있는 계 ($S'$ 계)에서 측정한 시간 간격으로, 가장 짧은 시간입니다.
    * **$S$ 계에서 측정한 시간**: $S$ 계의 관찰자가 이 두 사건 사이의 시간 간격을 측정한 것으로, $S'$ 계의 고유 시간보다 길어집니다.
      * $S$ 계에서 측정한 시간 $\Delta t_S = \gamma \Delta t_0$ 입니다.
      * 현재 $v={v_speed:.2f}C$ 이므로 $\gamma = {lorentz_factor(v_speed):.2f}$ 이고, $\Delta t_S = {1.0 * lorentz_factor(v_speed):.2f}$ 입니다.
      * 도표에서 보라색 선의 $t$축 투영 길이가 $1.0$보다 길어진 것을 확인할 수 있습니다.
    """)

st.markdown("""
---
**참고**: 이 도표는 (x, t) 2차원 시공간을 시각화합니다. 실제 시공간은 (x, y, z, t) 4차원입니다.
""")
