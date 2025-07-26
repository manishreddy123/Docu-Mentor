import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import streamlit as st
import base64
import io
import os


def show_pdf_preview(file_path, target_page=None, highlight_texts=None):
    """
    Display PDF preview with optional highlighting and target page navigation.
    Uses pdf2image for rendering and pdfplumber for text extraction.
    """
    if not os.path.exists(file_path):
        st.error(f"📄 Document not found: {os.path.basename(file_path)}")
        st.info("This document may have been moved or deleted. Please upload it again or delete this session.")
        return
        
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            max_pages_to_render = 10
            pages_to_render = min(max_pages_to_render, total_pages)
            
            pages = []
            
            for i in range(pages_to_render):
                page_number = i + 1
                
                images = convert_from_path(file_path, dpi=150, first_page=page_number, last_page=page_number, poppler_path="poppler/Library/bin")
                img = images[0]
                
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_data = img_buffer.getvalue()
                b64_img = base64.b64encode(img_data).decode()
                
                page_style = "margin-bottom: 15px; border-radius: 12px; max-width: 100%; box-shadow: 0 4px 12px rgba(147, 197, 253, 0.1);"
                if target_page and (page_number == target_page):
                    page_style += " border: 3px solid #1E3A8A; box-shadow: 0 0 20px rgba(30, 58, 138, 0.3);"
                
                pages.append(f'<img id="{page_number}" src="data:image/png;base64,{b64_img}" style="{page_style}" />')
            
            if total_pages > max_pages_to_render:
                pages.append(f'<p style="text-align: center; color: #666;">Showing first {max_pages_to_render} of {total_pages} pages</p>')
            
            st.markdown(
                f"<div style='height: 600px; overflow-y: auto; border: 2px solid #93C5FD; padding: 16px; border-radius: 12px; background: #FAFAFA;'>{''.join(pages)}</div>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        print(f"⚠️ PDF preview failed: {str(e)}")


def pdf_viewer(pdf_path, scroll_to_page=None, highlights=[]):
    """
    Alternative PDF viewer function for different use cases.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages = []
            
            for i in range(total_pages):
                page_number = i + 1
                
                images = convert_from_path(pdf_path, dpi=150, first_page=page_number, last_page=page_number)
                img = images[0]
                
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_data = img_buffer.getvalue()
                b64_img = base64.b64encode(img_data).decode()
                
                page_html = f'<img src="data:image/png;base64,{b64_img}" width="100%" style="margin-bottom: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(147, 197, 253, 0.1);"/>'
                if scroll_to_page == page_number:
                    page_html = f'<div style="border: 3px solid #1E3A8A; border-radius: 12px; box-shadow: 0 0 20px rgba(30, 58, 138, 0.3);">{page_html}</div>'
                pages.append(page_html)
            
            st.markdown(
                f"""
                <div style="height: 600px; overflow-y: auto; border: 2px solid #93C5FD; padding: 16px; border-radius: 12px; background: #FAFAFA;">
                    {"".join(pages)}
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        print(f"⚠️ PDF preview failed: {str(e)}")
