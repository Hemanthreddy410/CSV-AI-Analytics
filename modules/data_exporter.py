import streamlit as st
import pandas as pd
import io
import base64
import json
import datetime

class DataExporter:
    """Class for exporting data to various file formats"""
    
    def __init__(self, df=None):
        """Initialize with optional dataframe"""
        self.df = df
    
    def set_dataframe(self, df):
        """Set the dataframe to export"""
        self.df = df
    
    def render_export_options(self):
        """Render export options interface"""
        if self.df is None or self.df.empty:
            st.warning("No data available to export.")
            return
        
        st.subheader("Export Data")
        
        # Format selection
        export_format = st.selectbox(
            "Export format:",
            ["CSV", "Excel", "JSON", "HTML", "Pickle"]
        )
        
        # Additional options per format
        if export_format == "CSV":
            # CSV options
            csv_sep = st.selectbox("Separator:", [",", ";", "\t", "|"])
            csv_encoding = st.selectbox("Encoding:", ["utf-8", "latin1", "utf-16"])
            csv_index = st.checkbox("Include index", value=False)
            
            # Generate file
            if st.button("Generate CSV for Download", use_container_width=True):
                try:
                    csv_buffer = io.StringIO()
                    self.df.to_csv(csv_buffer, sep=csv_sep, encoding=csv_encoding, index=csv_index)
                    csv_str = csv_buffer.getvalue()
                    
                    # Create download button
                    st.download_button(
                        label="Download CSV File",
                        data=csv_str,
                        file_name=f"data_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error exporting to CSV: {str(e)}")
        
        elif export_format == "Excel":
            # Excel options
            excel_sheet = st.text_input("Sheet name:", "Sheet1")
            excel_index = st.checkbox("Include index", value=False, key="excel_idx")
            
            # Generate file
            if st.button("Generate Excel for Download", use_container_width=True):
                try:
                    excel_buffer = io.BytesIO()
                    self.df.to_excel(excel_buffer, sheet_name=excel_sheet, index=excel_index, engine="openpyxl")
                    excel_data = excel_buffer.getvalue()
                    
                    # Create download button
                    st.download_button(
                        label="Download Excel File",
                        data=excel_data,
                        file_name=f"data_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error exporting to Excel: {str(e)}")
        
        elif export_format == "JSON":
            # JSON options
            json_orient = st.selectbox(
                "JSON orientation:",
                ["records", "columns", "index", "split", "table"]
            )
            json_lines = st.checkbox("Lines format (records only)", value=False)
            
            # Generate file
            if st.button("Generate JSON for Download", use_container_width=True):
                try:
                    if json_lines and json_orient == "records":
                        json_str = self.df.to_json(orient="records", lines=True)
                    else:
                        json_str = self.df.to_json(orient=json_orient)
                    
                    # Create download button
                    st.download_button(
                        label="Download JSON File",
                        data=json_str,
                        file_name=f"data_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error exporting to JSON: {str(e)}")
        
        elif export_format == "HTML":
            # HTML options
            html_index = st.checkbox("Include index", value=True, key="html_idx")
            html_classes = st.text_input("CSS classes:", "dataframe table table-striped")
            
            # Generate file
            if st.button("Generate HTML for Download", use_container_width=True):
                try:
                    html_str = self.df.to_html(index=html_index, classes=html_classes.split())
                    
                    # Add basic styling
                    styled_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="UTF-8">
                        <title>Data Export</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            .dataframe {{ border-collapse: collapse; width: 100%; }}
                            .dataframe th, .dataframe td {{ 
                                border: 1px solid #ddd; 
                                padding: 8px; 
                                text-align: left;
                            }}
                            .dataframe th {{ 
                                background-color: #f2f2f2; 
                                color: #333;
                            }}
                            .dataframe tr:nth-child(even) {{ background-color: #f9f9f9; }}
                            .dataframe tr:hover {{ background-color: #eef; }}
                        </style>
                    </head>
                    <body>
                        <h1>Exported Data</h1>
                        <p>Exported on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        {html_str}
                    </body>
                    </html>
                    """
                    
                    # Create download button
                    st.download_button(
                        label="Download HTML File",
                        data=styled_html,
                        file_name=f"data_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error exporting to HTML: {str(e)}")
        
        elif export_format == "Pickle":
            # Pickle options
            pickle_protocol = st.slider("Pickle protocol version:", 0, 5, 4)
            pickle_compression = st.selectbox("Compression:", ["None", "gzip", "bz2", "xz"])
            
            # Generate file
            if st.button("Generate Pickle for Download", use_container_width=True):
                try:
                    pickle_buffer = io.BytesIO()
                    
                    if pickle_compression == "None":
                        self.df.to_pickle(pickle_buffer, protocol=pickle_protocol)
                    else:
                        self.df.to_pickle(
                            pickle_buffer, 
                            protocol=pickle_protocol,
                            compression=pickle_compression
                        )
                    
                    pickle_data = pickle_buffer.getvalue()
                    
                    # Create download button
                    file_extension = ".pkl" if pickle_compression == "None" else f".{pickle_compression}.pkl"
                    st.download_button(
                        label="Download Pickle File",
                        data=pickle_data,
                        file_name=f"data_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{file_extension}",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error exporting to Pickle: {str(e)}")
    
    def quick_export_widget(self, label="Download Data"):
        """Create a quick export widget for CSV download"""
        if self.df is None or self.df.empty:
            return False
        
        # Generate CSV
        csv = self.df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="data_export_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" class="download-button" style="text-decoration:none;">{label}</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        return True
    
    def export_to_file(self, file_path, format="csv", **kwargs):
        """Export dataframe to a file programmatically"""
        if self.df is None or self.df.empty:
            return False
        
        try:
            if format.lower() == "csv":
                self.df.to_csv(file_path, **kwargs)
            elif format.lower() in ["excel", "xlsx", "xls"]:
                self.df.to_excel(file_path, **kwargs)
            elif format.lower() == "json":
                self.df.to_json(file_path, **kwargs)
            elif format.lower() == "html":
                self.df.to_html(file_path, **kwargs)
            elif format.lower() == "pickle":
                self.df.to_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
        except Exception as e:
            print(f"Error exporting to file: {str(e)}")
            return False