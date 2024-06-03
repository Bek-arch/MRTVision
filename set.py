import base64
import plotly.graph_objects as go
import streamlit as st

def set_background(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    encoded_img = base64.b64encode(img_data).decode()
    background_style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_img});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

def visualize(image, bounding_boxes):
  
    img_width, img_height = image.size
    shapes = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        shapes.append(dict(
            type="rect",
            x0=x1,
            y0=img_height - y2,
            x1=x2,
            y1=img_height - y1,
            line=dict(color='red', width=6),
        ))

    fig = go.Figure()
    fig.update_layout(
        images=[dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=img_height,
            sizex=img_width,
            sizey=img_height,
            sizing="stretch"
        )]
    )

    fig.update_xaxes(range=[0, img_width], showticklabels=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[0, img_height], showticklabels=False)

    fig.update_layout(
        height=800,
        updatemenus=[
            dict(
                direction='left',
                pad=dict(r=10, t=10),
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
                type="buttons",
                buttons=[
                    dict(label="Original",
                         method="relayout",
                         args=["shapes", []]),
                    dict(label="Detections",
                         method="relayout",
                         args=["shapes", shapes])
                ],
            )
        ]
    )

    st.plotly_chart(fig)
