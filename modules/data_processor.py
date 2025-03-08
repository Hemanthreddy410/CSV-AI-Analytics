start_date_pd = pd.Timestamp(start_date)
                            end_date_pd = pd.Timestamp(end_date)
                            self.df = self.df[(self.df[filter_col] >= start_date_pd) & (self.df[filter_col] <= end_date_pd)]
                            filter_desc = f"{filter_col} between {start_date} and {end_date}"
                        elif filter_type == "After date":
                            filter_date_pd = pd.Timestamp(filter_date)
                            self.df = self.df[self.df[filter_col] > filter_date_pd]
                            filter_desc = f"{filter_col} > {filter_date}"
                        elif filter_type == "Before date":
                            filter_date_pd = pd.Timestamp(filter_date)
                            self.df = self.df[self.df[filter_col] < filter_date_pd]
                            filter_desc = f"{filter_col} < {filter_date}"
                        elif filter_type == "Equal to date":
                            filter_date_pd = pd.Timestamp(filter_date)
                            self.df = self.df[self.df[filter_col].dt.date == filter_date_pd.date()]
                            filter_desc = f"{filter_col} == {filter_date}"
                    
                    else:
                        # Boolean or other type
                        self.df = self.df[self.df[filter_col] == filter_value]
                        filter_desc = f"{filter_col} == {filter_value}"
                    
                    # Calculate rows removed
                    rows_removed = orig_shape[0] - self.df.shape[0]
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Filtered data where {filter_desc}",
                        "timestamp": datetime.datetime.now(),
                        "type": "filtering",
                        "details": {
                            "filter": filter_desc,
                            "rows_before": orig_shape[0],
                            "rows_after": self.df.shape[0],
                            "rows_removed": rows_removed
                        }
                    })
                    
                    st.success(f"Applied filter: {filter_desc} (Removed {rows_removed} rows, {self.df.shape[0]} remaining)")
                    
                    # Update the dataframe in session state
                    st.session_state.df = self.df
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error applying filter: {str(e)}")
        
        with col2:
            st.markdown("### Sampling")
            
            # Sample type
            sample_type = st.selectbox(
                "Sampling method:",
                ["Random Sample", "Stratified Sample", "Systematic Sample"],
                key="sample_type"
            )
            
            # Sample size
            sample_size_type = st.radio(
                "Sample size type:",
                ["Number of rows", "Percentage of data"],
                key="sample_size_type"
            )
            
            if sample_size_type == "Number of rows":
                sample_size = st.number_input(
                    "Number of rows:",
                    min_value=1,
                    max_value=len(self.df),
                    value=min(1000, len(self.df)),
                    key="sample_size_num"
                )
            else:
                sample_pct = st.slider(
                    "Percentage of data:",
                    min_value=1,
                    max_value=100,
                    value=20,
                    key="sample_size_pct"
                )
                sample_size = int(len(self.df) * (sample_pct / 100))
            
            # Additional options for stratified sampling
            if sample_type == "Stratified Sample":
                strat_col = st.selectbox(
                    "Stratify by column:",
                    self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist(),
                    key="strat_col"
                )
            
            # Apply sampling button
            if st.button("Apply Sampling", key="apply_sample"):
                try:
                    # Store original shape for reporting
                    orig_shape = self.df.shape
                    
                    # Apply the sampling based on type
                    if sample_type == "Random Sample":
                        self.df = self.df.sample(n=sample_size, random_state=42)
                        sample_desc = f"Random sample of {sample_size} rows"
                    
                    elif sample_type == "Stratified Sample":
                        try:
                            from sklearn.model_selection import train_test_split
                            
                            # Calculate proportion to maintain class distribution
                            prop = sample_size / len(self.df)
                            
                            # Split the data, keeping the sampled portion
                            _, sampled_df = train_test_split(
                                self.df,
                                test_size=prop,
                                stratify=self.df[strat_col],
                                random_state=42
                            )
                            
                            self.df = sampled_df
                            sample_desc = f"Stratified sample of {len(self.df)} rows (stratified by {strat_col})"
                            
                        except Exception as e:
                            st.error(f"Error in stratified sampling: {str(e)}")
                            st.info("Falling back to random sampling")
                            self.df = self.df.sample(n=sample_size, random_state=42)
                            sample_desc = f"Random sample of {sample_size} rows (fallback from stratified)"
                    
                    elif sample_type == "Systematic Sample":
                        # Calculate step size
                        step = len(self.df) // sample_size
                        
                        # Apply systematic sampling
                        indices = np.arange(0, len(self.df), step)[:sample_size]
                        self.df = self.df.iloc[indices]
                        
                        sample_desc = f"Systematic sample of {len(self.df)} rows (every {step}th row)"
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Applied {sample_desc}",
                        "timestamp": datetime.datetime.now(),
                        "type": "sampling",
                        "details": {
                            "method": sample_type,
                            "size": len(self.df),
                            "original_size": orig_shape[0]
                        }
                    })
                    
                    st.success(f"Applied {sample_desc}")
                    
                    # Update the dataframe in session state
                    st.session_state.df = self.df
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error applying sampling: {str(e)}")
            
            # Show dataset shape info
            st.markdown("### Current Dataset Info")
            st.metric("Rows", len(self.df))
            st.metric("Columns", len(self.df.columns))
            
            # Option to sample dataset before first analysis
            st.markdown("### Quick Row Subset")
            
            subset_options = ["All Data"]
            if len(self.df) > 1000:
                subset_options.extend(["First 1000 rows", "Last 1000 rows", "Random 1000 rows"])
            
            subset_choice = st.selectbox(
                "Quick subset for analysis:",
                subset_options,
                key="quick_subset"
            )
            
            if subset_choice != "All Data" and st.button("Apply Quick Subset", key="apply_quick_subset"):
                try:
                    # Store original shape for reporting
                    orig_shape = self.df.shape
                    
                    # Apply the subset
                    if subset_choice == "First 1000 rows":
                        self.df = self.df.head(1000)
                        subset_desc = "First 1000 rows"
                    elif subset_choice == "Last 1000 rows":
                        self.df = self.df.tail(1000)
                        subset_desc = "Last 1000 rows"
                    elif subset_choice == "Random 1000 rows":
                        self.df = self.df.sample(n=min(1000, len(self.df)), random_state=42)
                        subset_desc = "Random 1000 rows"
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Applied quick subset: {subset_desc}",
                        "timestamp": datetime.datetime.now(),
                        "type": "subset",
                        "details": {
                            "method": subset_choice,
                            "size": len(self.df),
                            "original_size": orig_shape[0]
                        }
                    })
                    
                    st.success(f"Applied quick subset: {subset_desc}")
                    
                    # Update the dataframe in session state
                    st.session_state.df = self.df
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error applying quick subset: {str(e)}")
    
    def _render_column_management(self):
        """Render column management interface"""
        st.subheader("Column Management")
        
        # Create columns for organized layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Rename Columns")
            
            # Select column to rename
            old_col = st.selectbox("Select column to rename:", self.df.columns, key="rename_old_col")
            new_col = st.text_input("New column name:", key="rename_new_col")
            
            if st.button("Rename Column", key="rename_col_btn"):
                try:
                    if not new_col:
                        st.error("Please provide a new column name")
                    elif new_col in self.df.columns:
                        st.error(f"Column name '{new_col}' already exists")
                    else:
                        # Create a copy of the column mapping
                        new_columns = self.df.columns.tolist()
                        idx = new_columns.index(old_col)
                        new_columns[idx] = new_col
                        
                        # Rename the column
                        self.df.columns = new_columns
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Renamed column '{old_col}' to '{new_col}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "column_management",
                            "details": {
                                "operation": "rename",
                                "old_name": old_col,
                                "new_name": new_col
                            }
                        })
                        
                        st.success(f"Renamed column '{old_col}' to '{new_col}'")
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.experimental_rerun()
                        
                except Exception as e:
                    st.error(f"Error renaming column: {str(e)}")
            
            st.markdown("### Drop Columns")
            
            # Select columns to drop
            cols_to_drop = st.multiselect("Select columns to drop:", self.df.columns, key="drop_cols")
            
            if st.button("Drop Columns", key="drop_cols_btn"):
                try:
                    if not cols_to_drop:
                        st.error("Please select at least one column to drop")
                    else:
                        # Drop the columns
                        self.df = self.df.drop(columns=cols_to_drop)
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Dropped {len(cols_to_drop)} columns: {', '.join(cols_to_drop)}",
                            "timestamp": datetime.datetime.now(),
                            "type": "column_management",
                            "details": {
                                "operation": "drop",
                                "columns": cols_to_drop
                            }
                        })
                        
                        st.success(f"Dropped {len(cols_to_drop)} columns")
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.experimental_rerun()
                        
                except Exception as e:
                    st.error(f"Error dropping columns: {str(e)}")
            
            st.markdown("### Change Column Data Type")
            
            # Select column to change type
            col_to_change = st.selectbox("Select column to change type:", self.df.columns, key="change_type_col")
            
            # Get current type
            current_type = str(self.df[col_to_change].dtype)
            
            # Available types
            new_type = st.selectbox(
                f"Change type from {current_type} to:",
                ["int", "float", "str", "bool", "datetime", "category"],
                key="new_type"
            )
            
            if st.button("Change Data Type", key="change_type_btn"):
                try:
                    # Change the data type
                    if new_type == "int":
                        self.df[col_to_change] = pd.to_numeric(self.df[col_to_change], errors='coerce').fillna(0).astype(int)
                    elif new_type == "float":
                        self.df[col_to_change] = pd.to_numeric(self.df[col_to_change], errors='coerce')
                    elif new_type == "str":
                        self.df[col_to_change] = self.df[col_to_change].astype(str)
                    elif new_type == "bool":
                        if pd.api.types.is_numeric_dtype(self.df[col_to_change]):
                            self.df[col_to_change] = self.df[col_to_change] != 0
                        else:
                            self.df[col_to_change] = self.df[col_to_change].astype(str).str.lower().isin(['true', 'yes', 'y', '1', 't'])
                    elif new_type == "datetime":
                        self.df[col_to_change] = pd.to_datetime(self.df[col_to_change], errors='coerce')
                    elif new_type == "category":
                        self.df[col_to_change] = self.df[col_to_change].astype('category')
                    
                    # Add to processing history
                    st.session_state.processing_history.append({
                        "description": f"Changed data type of '{col_to_change}' from {current_type} to {new_type}",
                        "timestamp": datetime.datetime.now(),
                        "type": "column_management",
                        "details": {
                            "operation": "change_type",
                            "column": col_to_change,
                            "old_type": current_type,
                            "new_type": new_type
                        }
                    })
                    
                    st.success(f"Changed data type of '{col_to_change}' to {new_type}")
                    
                    # Update the dataframe in session state
                    st.session_state.df = self.df
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error changing data type: {str(e)}")
        
        with col2:
            st.markdown("### Reorder Columns")
            
            # Show current column order
            st.write("Current column order:")
            current_cols = self.df.columns.tolist()
            for i, col in enumerate(current_cols[:10]):  # Show first 10 columns
                st.text(f"{i+1}. {col}")
            if len(current_cols) > 10:
                st.text(f"... and {len(current_cols)-10} more columns")
            
            st.write("Choose reordering method:")
            reorder_method = st.radio(
                "Reordering method:",
                ["Move column to position", "Alphabetical order", "Custom order"],
                key="reorder_method"
            )
            
            if reorder_method == "Move column to position":
                col_to_move = st.selectbox("Select column to move:", current_cols, key="move_col")
                new_position = st.number_input(
                    "Move to position (1-based):",
                    min_value=1,
                    max_value=len(current_cols),
                    value=1,
                    key="new_position"
                )
                
                if st.button("Move Column", key="move_col_btn"):
                    try:
                        # Get current position
                        current_pos = current_cols.index(col_to_move)
                        
                        # Create a new column order
                        new_columns = current_cols.copy()
                        new_columns.pop(current_pos)
                        new_columns.insert(new_position - 1, col_to_move)
                        
                        # Reorder columns
                        self.df = self.df[new_columns]
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Moved column '{col_to_move}' to position {new_position}",
                            "timestamp": datetime.datetime.now(),
                            "type": "column_management",
                            "details": {
                                "operation": "reorder",
                                "method": "move",
                                "column": col_to_move,
                                "new_position": new_position
                            }
                        })
                        
                        st.success(f"Moved column '{col_to_move}' to position {new_position}")
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error moving column: {str(e)}")
            
            elif reorder_method == "Alphabetical order":
                sort_direction = st.radio(
                    "Sort direction:",
                    ["Ascending (A-Z)", "Descending (Z-A)"],
                    key="sort_direction"
                )
                
                if st.button("Sort Columns", key="sort_cols_btn"):
                    try:
                        # Sort columns
                        sorted_cols = sorted(current_cols, reverse=(sort_direction == "Descending (Z-A)"))
                        
                        # Reorder columns
                        self.df = self.df[sorted_cols]
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Sorted columns in {sort_direction.lower()} order",
                            "timestamp": datetime.datetime.now(),
                            "type": "column_management",
                            "details": {
                                "operation": "reorder",
                                "method": "sort",
                                "direction": sort_direction
                            }
                        })
                        
                        st.success(f"Sorted columns in {sort_direction.lower()} order")
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error sorting columns: {str(e)}")
            
            elif reorder_method == "Custom order":
                st.write("Enter comma-separated list of column names in desired order:")
                st.write("(You can copy-paste from the current columns shown above)")
                custom_order = st.text_area("Column order:", key="custom_order")
                
                if st.button("Apply Custom Order", key="custom_order_btn"):
                    try:
                        # Parse custom order
                        new_columns = [col.strip() for col in custom_order.split(',')]
                        
                        # Check if all columns are included
                        missing_cols = set(current_cols) - set(new_columns)
                        extra_cols = set(new_columns) - set(current_cols)
                        
                        if extra_cols:
                            st.error(f"Invalid column names: {', '.join(extra_cols)}")
                        elif missing_cols:
                            st.warning(f"Missing columns: {', '.join(missing_cols)}")
                            st.info("Append missing columns to the end?")
                            
                            if st.button("Append Missing Columns", key="append_missing"):
                                new_columns.extend(missing_cols)
                        
                        # Apply reordering
                        self.df = self.df[new_columns]
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Applied custom column order",
                            "timestamp": datetime.datetime.now(),
                            "type": "column_management",
                            "details": {
                                "operation": "reorder",
                                "method": "custom",
                                "new_order": new_columns
                            }
                        })
                        
                        st.success(f"Applied custom column order")
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying custom order: {str(e)}")
            
            st.markdown("### Create Column Copy")
            
            # Select column to copy
            col_to_copy = st.selectbox("Select column to copy:", self.df.columns, key="copy_col")
            new_col_name = st.text_input("New column name:", key="copy_new_name")
            
            if st.button("Create Copy", key="copy_col_btn"):
                try:
                    if not new_col_name:
                        st.error("Please provide a name for the new column")
                    elif new_col_name in self.df.columns:
                        st.error(f"Column name '{new_col_name}' already exists")
                    else:
                        # Create copy
                        self.df[new_col_name] = self.df[col_to_copy].copy()
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Created copy of column '{col_to_copy}' as '{new_col_name}'",
                            "timestamp": datetime.datetime.now(),
                            "type": "column_management",
                            "details": {
                                "operation": "copy",
                                "original_column": col_to_copy,
                                "new_column": new_col_name
                            }
                        })
                        
                        st.success(f"Created copy of column '{col_to_copy}' as '{new_col_name}'")
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.experimental_rerun()
                        
                except Exception as e:
                    st.error(f"Error creating column copy: {str(e)}")
                                    self.df[new_col_name] = self.df[col].str.upper()
                                    operation_desc = "uppercase"
                                elif operation == "To Lowercase":
                                    self.df[new_col_name] = self.df[col].str.lower()
                                    operation_desc = "lowercase"
                                elif operation == "Extract Substring":
                                    self.df[new_col_name] = self.df[col].str[start_idx:end_idx]
                                    operation_desc = f"substring({start_idx}:{end_idx})"
                                elif operation == "String Length":
                                    self.df[new_col_name] = self.df[col].str.len()
                                    operation_desc = "length"
                                elif operation == "Replace Text":
                                    if not old_text:
                                        st.error("Please provide text to replace")
                                        return
                                    self.df[new_col_name] = self.df[col].str.replace(old_text, new_text)
                                    operation_desc = f"replace('{old_text}' with '{new_text}')"
                                elif operation == "Remove Whitespace":
                                    self.df[new_col_name] = self.df[col].str.strip()
                                    operation_desc = "whitespace removal"
                                elif operation == "Extract Pattern (Regex)":
                                    if not pattern:
                                        st.error("Please provide a regex pattern")
                                        return
                                    self.df[new_col_name] = self.df[col].str.extract(f"({pattern})", expand=False)
                                    operation_desc = f"regex extract('{pattern}')"
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Created feature '{new_col_name}' using {operation_desc} on '{col}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "feature_engineering",
                                    "details": {
                                        "method": "string_operation",
                                        "column": col,
                                        "operation": operation,
                                        "new_column": new_col_name
                                    }
                                })
                                
                                st.success(f"Created new column '{new_col_name}'")
                                
                                # Update the dataframe in session state
                                st.session_state.df = self.df
                                st.experimental_rerun()
                                
                        except Exception as e:
                            st.error(f"Error creating feature: {str(e)}")
            
            elif feature_method == "Conditional Logic":
                # Create a new feature based on conditions
                
                # Get columns
                all_cols = self.df.columns.tolist()
                
                if not all_cols:
                    st.warning("Conditional logic requires at least one column.")
                else:
                    col = st.selectbox("Select column for condition:", all_cols, key="cond_col")
                    
                    condition_type = st.selectbox(
                        "Condition type:",
                        [
                            "Greater than",
                            "Less than",
                            "Equal to",
                            "Not equal to",
                            "Contains text",
                            "Starts with",
                            "Ends with",
                            "Is in list"
                        ],
                        key="cond_type"
                    )
                    
                    # Condition value depends on the type of condition
                    if condition_type in ["Contains text", "Starts with", "Ends with"]:
                        cond_value = st.text_input("Condition value:", key="cond_text_value")
                    elif condition_type == "Is in list":
                        cond_value = st.text_input("Comma-separated list of values:", key="cond_list_value")
                    else:
                        # For numeric conditions
                        try:
                            if pd.api.types.is_numeric_dtype(self.df[col]):
                                # Suggest a reasonable default
                                default_val = self.df[col].mean()
                            else:
                                default_val = 0
                            cond_value = st.number_input("Condition value:", value=default_val, key="cond_num_value")
                        except:
                            cond_value = st.text_input("Condition value:", key="cond_general_value")
                    
                    # Values for true and false conditions
                    true_value = st.text_input("Value if condition is true:", key="true_value")
                    false_value = st.text_input("Value if condition is false:", key="false_value")
                    
                    new_col_name = st.text_input("New column name:", key="cond_new_col")
                    
                    if st.button("Create Feature", key="create_cond"):
                        try:
                            if not new_col_name:
                                st.error("Please provide a name for the new column")
                            elif not true_value or not false_value:
                                st.error("Please provide values for both true and false conditions")
                            else:
                                # Try to convert true/false values to appropriate types
                                try:
                                    if pd.api.types.is_numeric_dtype(self.df[col]):
                                        true_val = float(true_value) if '.' in true_value else int(true_value)
                                        false_val = float(false_value) if '.' in false_value else int(false_value)
                                    else:
                                        true_val = true_value
                                        false_val = false_value
                                except:
                                    true_val = true_value
                                    false_val = false_value
                                
                                # Apply the condition
                                if condition_type == "Greater than":
                                    self.df[new_col_name] = np.where(self.df[col] > cond_value, true_val, false_val)
                                    condition_desc = f"{col} > {cond_value}"
                                elif condition_type == "Less than":
                                    self.df[new_col_name] = np.where(self.df[col] < cond_value, true_val, false_val)
                                    condition_desc = f"{col} < {cond_value}"
                                elif condition_type == "Equal to":
                                    self.df[new_col_name] = np.where(self.df[col] == cond_value, true_val, false_val)
                                    condition_desc = f"{col} == {cond_value}"
                                elif condition_type == "Not equal to":
                                    self.df[new_col_name] = np.where(self.df[col] != cond_value, true_val, false_val)
                                    condition_desc = f"{col} != {cond_value}"
                                elif condition_type == "Contains text":
                                    self.df[new_col_name] = np.where(self.df[col].str.contains(cond_value, na=False), true_val, false_val)
                                    condition_desc = f"{col} contains '{cond_value}'"
                                elif condition_type == "Starts with":
                                    self.df[new_col_name] = np.where(self.df[col].str.startswith(cond_value, na=False), true_val, false_val)
                                    condition_desc = f"{col} starts with '{cond_value}'"
                                elif condition_type == "Ends with":
                                    self.df[new_col_name] = np.where(self.df[col].str.endswith(cond_value, na=False), true_val, false_val)
                                    condition_desc = f"{col} ends with '{cond_value}'"
                                elif condition_type == "Is in list":
                                    value_list = [v.strip() for v in cond_value.split(',')]
                                    self.df[new_col_name] = np.where(self.df[col].isin(value_list), true_val, false_val)
                                    condition_desc = f"{col} in [{cond_value}]"
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Created feature '{new_col_name}' where {condition_desc}",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "feature_engineering",
                                    "details": {
                                        "method": "conditional_logic",
                                        "condition": condition_desc,
                                        "true_value": true_val,
                                        "false_value": false_val,
                                        "new_column": new_col_name
                                    }
                                })
                                
                                st.success(f"Created new column '{new_col_name}'")
                                
                                # Update the dataframe in session state
                                st.session_state.df = self.df
                                st.experimental_rerun()
                                
                        except Exception as e:
                            st.error(f"Error creating feature: {str(e)}")
                            
            elif feature_method == "Aggregation by Group":
                # Create aggregated features by grouping
                
                # Get columns
                all_cols = self.df.columns.tolist()
                num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
                cat_cols = [col for col in all_cols if col not in num_cols]
                
                if not num_cols or not cat_cols:
                    st.warning("Aggregation requires at least one numeric column and one categorical column.")
                else:
                    group_col = st.selectbox("Group by column:", cat_cols, key="agg_group_col")
                    agg_col = st.selectbox("Column to aggregate:", num_cols, key="agg_data_col")
                    
                    agg_function = st.selectbox(
                        "Aggregation function:",
                        ["mean", "median", "sum", "min", "max", "count", "std", "var"],
                        key="agg_func"
                    )
                    
                    new_col_name = st.text_input("New column name:", key="agg_new_col")
                    
                    if st.button("Create Feature", key="create_agg"):
                        try:
                            if not new_col_name:
                                st.error("Please provide a name for the new column")
                            else:
                                # Calculate aggregation
                                agg_values = self.df.groupby(group_col)[agg_col].agg(agg_function)
                                
                                # Map values back to original dataframe
                                self.df[new_col_name] = self.df[group_col].map(agg_values)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Created feature '{new_col_name}' using {agg_function} of {agg_col} grouped by {group_col}",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "feature_engineering",
                                    "details": {
                                        "method": "aggregation",
                                        "group_column": group_col,
                                        "agg_column": agg_col,
                                        "function": agg_function,
                                        "new_column": new_col_name
                                    }
                                })
                                
                                st.success(f"Created new column '{new_col_name}'")
                                
                                # Update the dataframe in session state
                                st.session_state.df = self.df
                                st.experimental_rerun()
                                
                        except Exception as e:
                            st.error(f"Error creating feature: {str(e)}")
            
            elif feature_method == "Rolling Window":
                # Create rolling window features
                
                # Get numeric columns
                num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
                
                # Look for potential time/date columns
                date_cols = []
                for col in self.df.columns:
                    # Check if the column name suggests it's a date
                    if any(date_term in col.lower() for date_term in ['date', 'time', 'year', 'month', 'day']):
                        date_cols.append(col)
                    
                    # Or check if it's already a datetime type
                    elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                        date_cols.append(col)
                
                if not num_cols:
                    st.warning("Rolling window operations require at least one numeric column.")
                else:
                    col = st.selectbox("Select column for rolling window:", num_cols, key="roll_col")
                    window_size = st.number_input("Window size:", value=3, min_value=2, key="roll_window")
                    
                    function = st.selectbox(
                        "Aggregation function:",
                        ["mean", "median", "sum", "min", "max", "std"],
                        key="roll_func"
                    )
                    
                    # If date columns available, ask if sorting by a date column
                    sort_by_date = False
                    date_col = None
                    if date_cols:
                        sort_by_date = st.checkbox("Sort by date column", key="roll_sort_date")
                        if sort_by_date:
                            date_col = st.selectbox("Select date column for sorting:", date_cols, key="roll_date_col")
                    
                    new_col_name = st.text_input("New column name:", key="roll_new_col")
                    
                    if st.button("Create Feature", key="create_roll"):
                        try:
                            if not new_col_name:
                                st.error("Please provide a name for the new column")
                            else:
                                # Create a copy of the dataframe to avoid modifying the original during sorting
                                df_copy = self.df.copy()
                                
                                # Sort by date if requested
                                if sort_by_date and date_col:
                                    # Convert to datetime if needed
                                    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
                                        try:
                                            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                                        except:
                                            st.error(f"Could not convert '{date_col}' to datetime")
                                            return
                                    
                                    # Sort by date
                                    df_copy = df_copy.sort_values(date_col)
                                
                                # Calculate rolling window
                                if function == "mean":
                                    df_copy[new_col_name] = df_copy[col].rolling(window=window_size).mean()
                                elif function == "median":
                                    df_copy[new_col_name] = df_copy[col].rolling(window=window_size).median()
                                elif function == "sum":
                                    df_copy[new_col_name] = df_copy[col].rolling(window=window_size).sum()
                                elif function == "min":
                                    df_copy[new_col_name] = df_copy[col].rolling(window=window_size).min()
                                elif function == "max":
                                    df_copy[new_col_name] = df_copy[col].rolling(window=window_size).max()
                                elif function == "std":
                                    df_copy[new_col_name] = df_copy[col].rolling(window=window_size).std()
                                
                                # If we sorted, we need to restore the original order
                                if sort_by_date and date_col:
                                    # Keep just the new column and index
                                    result = df_copy[[new_col_name]]
                                    
                                    # Add the new column to the original dataframe
                                    self.df = self.df.join(result)
                                else:
                                    # Just add the new column directly
                                    self.df[new_col_name] = df_copy[new_col_name]
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Created feature '{new_col_name}' using rolling {function} with window size {window_size} on '{col}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "feature_engineering",
                                    "details": {
                                        "method": "rolling_window",
                                        "column": col,
                                        "window_size": window_size,
                                        "function": function,
                                        "sort_by": date_col if sort_by_date else None,
                                        "new_column": new_col_name
                                    }
                                })
                                
                                st.success(f"Created new column '{new_col_name}'")
                                
                                # Update the dataframe in session state
                                st.session_state.df = self.df
                                st.experimental_rerun()
                                
                        except Exception as e:
                            st.error(f"Error creating feature: {str(e)}")
        
        with col2:
            st.markdown("### Polynomial Features")
            
            # Get numeric columns
            num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if len(num_cols) < 1:
                st.info("Polynomial features require at least one numeric column.")
            else:
                # Select columns for polynomial features
                selected_cols = st.multiselect("Select columns for polynomial features:", num_cols, key="poly_cols")
                
                if selected_cols:
                    degree = st.slider("Polynomial degree:", 2, 5, 2, key="poly_degree")
                    interaction_only = st.checkbox("Interaction terms only (no powers)", key="poly_interaction_only")
                    include_bias = st.checkbox("Include bias (constant) term", key="poly_bias")
                    
                    if st.button("Generate Polynomial Features", key="create_poly"):
                        try:
                            from sklearn.preprocessing import PolynomialFeatures
                            
                            # Create polynomial features
                            poly = PolynomialFeatures(
                                degree=degree,
                                interaction_only=interaction_only,
                                include_bias=include_bias
                            )
                            
                            # Fit and transform selected columns
                            poly_features = poly.fit_transform(self.df[selected_cols])
                            
                            # Get feature names
                            feature_names = poly.get_feature_names_out(selected_cols)
                            
                            # Add polynomial features to dataframe
                            for i, name in enumerate(feature_names):
                                # Skip the bias term (constant 1) if it exists
                                if name == '1' and include_bias:
                                    continue
                                
                                # Replace ^ with _ for cleaner column names
                                clean_name = name.replace(' ', '_').replace('^', '_').replace('_1', '')
                                self.df[f"poly_{clean_name}"] = poly_features[:, i]
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Generated polynomial features (degree {degree}) for {', '.join(selected_cols)}",
                                "timestamp": datetime.datetime.now(),
                                "type": "feature_engineering",
                                "details": {
                                    "method": "polynomial",
                                    "columns": selected_cols,
                                    "degree": degree,
                                    "interaction_only": interaction_only,
                                    "include_bias": include_bias
                                }
                            })
                            
                            st.success(f"Created {len(feature_names) - (1 if include_bias else 0)} polynomial features")
                            
                            # Update the dataframe in session state
                            st.session_state.df = self.df
                            st.experimental_rerun()
                            
                        except Exception as e:
                            st.error(f"Error creating polynomial features: {str(e)}")
            
            st.markdown("### Interaction Terms")
            
            # Get numeric columns
            num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if len(num_cols) < 2:
                st.info("Interaction terms require at least two numeric columns.")
            else:
                col1 = st.selectbox("Select first column:", num_cols, key="interact_col1")
                col2_options = [col for col in num_cols if col != col1]
                col2 = st.selectbox("Select second column:", col2_options, key="interact_col2")
                
                interaction_type = st.selectbox(
                    "Interaction type:",
                    ["Multiplication", "Division", "Addition", "Subtraction"],
                    key="interact_type"
                )
                
                new_col_name = st.text_input("New column name (or leave blank for auto-name):", key="interact_new_col")
                
                if st.button("Create Interaction", key="create_interact"):
                    try:
                        # Auto-generate name if not provided
                        if not new_col_name:
                            if interaction_type == "Multiplication":
                                new_col_name = f"{col1}_mul_{col2}"
                                op_symbol = "*"
                            elif interaction_type == "Division":
                                new_col_name = f"{col1}_div_{col2}"
                                op_symbol = "/"
                            elif interaction_type == "Addition":
                                new_col_name = f"{col1}_add_{col2}"
                                op_symbol = "+"
                            elif interaction_type == "Subtraction":
                                new_col_name = f"{col1}_sub_{col2}"
                                op_symbol = "-"
                        
                        # Create interaction term
                        if interaction_type == "Multiplication":
                            self.df[new_col_name] = self.df[col1] * self.df[col2]
                        elif interaction_type == "Division":
                            # Handle division by zero
                            self.df[new_col_name] = self.df[col1] / self.df[col2].replace(0, np.nan)
                        elif interaction_type == "Addition":
                            self.df[new_col_name] = self.df[col1] + self.df[col2]
                        elif interaction_type == "Subtraction":
                            self.df[new_col_name] = self.df[col1] - self.df[col2]
                        
                        # Add to processing history
                        st.session_state.processing_history.append({
                            "description": f"Created interaction term '{new_col_name}' as {col1} {op_symbol} {col2}",
                            "timestamp": datetime.datetime.now(),
                            "type": "feature_engineering",
                            "details": {
                                "method": "interaction",
                                "column1": col1,
                                "column2": col2,
                                "operation": interaction_type,
                                "new_column": new_col_name
                            }
                        })
                        
                        st.success(f"Created interaction term '{new_col_name}'")
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error creating interaction term: {str(e)}")
    
    def _render_data_filtering(self):
        """Render data filtering interface"""
        st.subheader("Data Filtering")
        
        # Create columns for organized layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Filter Rows")
            
            # Provide dropdown for column selection
            filter_col = st.selectbox("Select column for filtering:", self.df.columns, key="filter_col")
            
            # Different filter options based on column data type
            col_dtype = self.df[filter_col].dtype
            
            if pd.api.types.is_numeric_dtype(col_dtype):
                # Numeric column
                filter_type = st.selectbox(
                    "Filter type:",
                    ["Range", "Greater than", "Less than", "Equal to", "Not equal to"],
                    key="num_filter_type"
                )
                
                if filter_type == "Range":
                    min_val = self.df[filter_col].min()
                    max_val = self.df[filter_col].max()
                    range_min, range_max = st.slider(
                        "Select range:",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=(float(min_val), float(max_val)),
                        key="range_slider"
                    )
                else:
                    filter_value = st.number_input(
                        "Filter value:",
                        value=float(self.df[filter_col].mean()),
                        key="num_filter_value"
                    )
            
            elif pd.api.types.is_categorical_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype):
                # Categorical column
                unique_values = self.df[filter_col].dropna().unique().tolist()
                
                filter_type = st.selectbox(
                    "Filter type:",
                    ["Include values", "Exclude values", "Contains text", "Starts with", "Ends with"],
                    key="cat_filter_type"
                )
                
                if filter_type in ["Include values", "Exclude values"]:
                    filter_values = st.multiselect(
                        "Select values:",
                        unique_values,
                        key="cat_filter_values"
                    )
                else:
                    filter_text = st.text_input(
                        "Enter text:",
                        key="text_filter_value"
                    )
            
            elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                # Datetime column
                filter_type = st.selectbox(
                    "Filter type:",
                    ["Date range", "After date", "Before date", "Equal to date"],
                    key="date_filter_type"
                )
                
                min_date = self.df[filter_col].min()
                max_date = self.df[filter_col].max()
                
                if filter_type == "Date range":
                    start_date = st.date_input(
                        "Start date:",
                        value=pd.to_datetime(min_date),
                        min_value=pd.to_datetime(min_date),
                        max_value=pd.to_datetime(max_date),
                        key="date_range_start"
                    )
                    end_date = st.date_input(
                        "End date:",
                        value=pd.to_datetime(max_date),
                        min_value=pd.to_datetime(min_date),
                        max_value=pd.to_datetime(max_date),
                        key="date_range_end"
                    )
                else:
                    filter_date = st.date_input(
                        "Select date:",
                        value=pd.to_datetime(min_date),
                        min_value=pd.to_datetime(min_date),
                        max_value=pd.to_datetime(max_date),
                        key="date_filter_value"
                    )
            
            else:
                # Boolean or other type
                filter_type = "Equal to"
                filter_value = st.selectbox(
                    "Filter value:",
                    [True, False] if pd.api.types.is_bool_dtype(col_dtype) else self.df[filter_col].unique().tolist(),
                    key="other_filter_value"
                )
            
            # Apply filter button
            if st.button("Apply Filter", key="apply_filter"):
                try:
                    # Store original shape for reporting
                    orig_shape = self.df.shape
                    
                    # Apply the filter based on type and column type
                    if pd.api.types.is_numeric_dtype(col_dtype):
                        if filter_type == "Range":
                            self.df = self.df[(self.df[filter_col] >= range_min) & (self.df[filter_col] <= range_max)]
                            filter_desc = f"{filter_col} between {range_min} and {range_max}"
                        elif filter_type == "Greater than":
                            self.df = self.df[self.df[filter_col] > filter_value]
                            filter_desc = f"{filter_col} > {filter_value}"
                        elif filter_type == "Less than":
                            self.df = self.df[self.df[filter_col] < filter_value]
                            filter_desc = f"{filter_col} < {filter_value}"
                        elif filter_type == "Equal to":
                            self.df = self.df[self.df[filter_col] == filter_value]
                            filter_desc = f"{filter_col} == {filter_value}"
                        elif filter_type == "Not equal to":
                            self.df = self.df[self.df[filter_col] != filter_value]
                            filter_desc = f"{filter_col} != {filter_value}"
                    
                    elif pd.api.types.is_categorical_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype):
                        if filter_type == "Include values":
                            if not filter_values:
                                st.error("Please select at least one value")
                                return
                            self.df = self.df[self.df[filter_col].isin(filter_values)]
                            filter_desc = f"{filter_col} in {filter_values}"
                        elif filter_type == "Exclude values":
                            if not filter_values:
                                st.error("Please select at least one value")
                                return
                            self.df = self.df[~self.df[filter_col].isin(filter_values)]
                            filter_desc = f"{filter_col} not in {filter_values}"
                        elif filter_type == "Contains text":
                            if not filter_text:
                                st.error("Please enter text to filter by")
                                return
                            self.df = self.df[self.df[filter_col].str.contains(filter_text, na=False)]
                            filter_desc = f"{filter_col} contains '{filter_text}'"
                        elif filter_type == "Starts with":
                            if not filter_text:
                                st.error("Please enter text to filter by")
                                return
                            self.df = self.df[self.df[filter_col].str.startswith(filter_text, na=False)]
                            filter_desc = f"{filter_col} starts with '{filter_text}'"
                        elif filter_type == "Ends with":
                            if not filter_text:
                                st.error("Please enter text to filter by")
                                return
                            self.df = self.df[self.df[filter_col].str.endswith(filter_text, na=False)]
                            filter_desc = f"{filter_col} ends with '{filter_text}'"
                    
                    elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                        if filter_type == "Date range":
                            start_date_pd = pd.Timestamp(start_date)
                            end_date_pd =                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied square root transform to '{col_to_transform}' with constant {const}",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "sqrt_transform",
                                        "constant": const,
                                        "new_column": f"{col_to_transform}_sqrt"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_sqrt' with square root transform (added constant {const})")
                            else:
                                self.df[f"{col_to_transform}_sqrt"] = np.sqrt(self.df[col_to_transform])
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied square root transform to '{col_to_transform}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "sqrt_transform",
                                        "new_column": f"{col_to_transform}_sqrt"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_sqrt' with square root transform")
                        
                        elif transform_method == "Box-Cox Transform":
                            # Check if all values are positive
                            if (self.df[col_to_transform] <= 0).any():
                                min_val = self.df[col_to_transform].min()
                                # Add a constant to make all values positive
                                const = abs(min_val) + 1 if min_val <= 0 else 0
                                
                                from scipy import stats
                                transformed_data, lambda_val = stats.boxcox(self.df[col_to_transform] + const)
                                self.df[f"{col_to_transform}_boxcox"] = transformed_data
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied Box-Cox transform to '{col_to_transform}' with constant {const}",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "boxcox_transform",
                                        "constant": const,
                                        "lambda": lambda_val,
                                        "new_column": f"{col_to_transform}_boxcox"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_boxcox' with Box-Cox transform (lambda={lambda_val:.4f}, added constant {const})")
                            else:
                                from scipy import stats
                                transformed_data, lambda_val = stats.boxcox(self.df[col_to_transform])
                                self.df[f"{col_to_transform}_boxcox"] = transformed_data
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied Box-Cox transform to '{col_to_transform}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "boxcox_transform",
                                        "lambda": lambda_val,
                                        "new_column": f"{col_to_transform}_boxcox"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_boxcox' with Box-Cox transform (lambda={lambda_val:.4f})")
                        
                        elif transform_method == "Binning/Discretization":
                            # Create bins
                            bins = pd.cut(
                                self.df[col_to_transform], 
                                bins=num_bins, 
                                labels=bin_labels.split(',') if bin_labels else None
                            )
                            
                            self.df[f"{col_to_transform}_binned"] = bins
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied binning to '{col_to_transform}' with {num_bins} bins",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "binning",
                                    "num_bins": num_bins,
                                    "new_column": f"{col_to_transform}_binned"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_binned' with {num_bins} bins")
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying transformation: {str(e)}")
        
        with col2:
            st.markdown("### Categorical Transformations")
            
            # Get categorical columns
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not cat_cols:
                st.info("No categorical columns found for transformation.")
            else:
                col_to_transform = st.selectbox(
                    "Select column to transform:",
                    cat_cols,
                    key="cat_transform_col"
                )
                
                transform_method = st.selectbox(
                    "Select transformation method:",
                    [
                        "One-Hot Encoding",
                        "Label Encoding",
                        "Frequency Encoding",
                        "Target Encoding (requires target column)"
                    ],
                    key="cat_transform_method"
                )
                
                # Additional parameters for specific transforms
                if transform_method == "Target Encoding (requires target column)":
                    target_col = st.selectbox(
                        "Select target column:",
                        self.df.columns
                    )
                
                # Apply button
                if st.button("Apply Transformation", key="apply_cat_transform"):
                    try:
                        if transform_method == "One-Hot Encoding":
                            # Get dummies (one-hot encoding)
                            dummies = pd.get_dummies(self.df[col_to_transform], prefix=col_to_transform)
                            
                            # Add to dataframe
                            self.df = pd.concat([self.df, dummies], axis=1)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied one-hot encoding to '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "one_hot_encoding",
                                    "new_columns": dummies.columns.tolist()
                                }
                            })
                            
                            st.success(f"Created {dummies.shape[1]} new columns with one-hot encoding for '{col_to_transform}'")
                        
                        elif transform_method == "Label Encoding":
                            # Map each unique value to an integer
                            unique_values = self.df[col_to_transform].dropna().unique()
                            mapping = {val: i for i, val in enumerate(unique_values)}
                            
                            self.df[f"{col_to_transform}_encoded"] = self.df[col_to_transform].map(mapping)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied label encoding to '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "label_encoding",
                                    "mapping": mapping,
                                    "new_column": f"{col_to_transform}_encoded"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_encoded' with label encoding")
                        
                        elif transform_method == "Frequency Encoding":
                            # Replace each category with its frequency
                            freq = self.df[col_to_transform].value_counts(normalize=True)
                            self.df[f"{col_to_transform}_freq"] = self.df[col_to_transform].map(freq)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Applied frequency encoding to '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "frequency_encoding",
                                    "new_column": f"{col_to_transform}_freq"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_freq' with frequency encoding")
                        
                        elif transform_method == "Target Encoding (requires target column)":
                            # Check if target column exists and is numeric
                            if target_col not in self.df.columns:
                                st.error(f"Target column '{target_col}' not found")
                            elif self.df[target_col].dtype.kind not in 'bifc':
                                st.error(f"Target column '{target_col}' must be numeric")
                            else:
                                # Calculate mean target value for each category
                                target_means = self.df.groupby(col_to_transform)[target_col].mean()
                                self.df[f"{col_to_transform}_target"] = self.df[col_to_transform].map(target_means)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied target encoding to '{col_to_transform}' using '{target_col}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "target_encoding",
                                        "target_column": target_col,
                                        "new_column": f"{col_to_transform}_target"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_target' with target encoding")
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying transformation: {str(e)}")
            
            # Date Transformations
            st.markdown("### Date Transformations")
            
            # Get columns that could be dates
            date_cols = []
            for col in self.df.columns:
                # Check if the column name suggests it's a date
                if any(date_term in col.lower() for date_term in ['date', 'time', 'year', 'month', 'day']):
                    date_cols.append(col)
                
                # Or check if it's already a datetime type
                elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    date_cols.append(col)
            
            if not date_cols:
                st.info("No date columns found for transformation.")
            else:
                col_to_transform = st.selectbox(
                    "Select date column to transform:",
                    date_cols,
                    key="date_transform_col"
                )
                
                # Check if we need to convert to datetime first
                needs_conversion = not pd.api.types.is_datetime64_any_dtype(self.df[col_to_transform])
                
                if needs_conversion:
                    st.warning(f"Column '{col_to_transform}' is not a datetime type. It will be converted first.")
                    date_format = st.text_input("Date format (e.g., '%Y-%m-%d'):", key="date_format")
                
                transform_method = st.selectbox(
                    "Select transformation method:",
                    [
                        "Extract Year",
                        "Extract Month",
                        "Extract Day",
                        "Extract Day of Week",
                        "Extract Hour",
                        "Extract Season",
                        "Days Since Reference Date"
                    ],
                    key="date_transform_method"
                )
                
                # Additional parameters for specific transforms
                if transform_method == "Days Since Reference Date":
                    ref_date = st.date_input("Reference date:", value=datetime.date.today())
                
                # Apply button
                if st.button("Apply Transformation", key="apply_date_transform"):
                    try:
                        # Convert to datetime if needed
                        if needs_conversion:
                            if not date_format:
                                st.error("Please specify a date format")
                                return
                            
                            self.df[f"{col_to_transform}_dt"] = pd.to_datetime(self.df[col_to_transform], format=date_format, errors='coerce')
                            date_col = f"{col_to_transform}_dt"
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Converted '{col_to_transform}' to datetime",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "datetime_conversion",
                                    "format": date_format,
                                    "new_column": date_col
                                }
                            })
                        else:
                            date_col = col_to_transform
                        
                        # Apply the selected transformation
                        if transform_method == "Extract Year":
                            self.df[f"{col_to_transform}_year"] = self.df[date_col].dt.year
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Extracted year from '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "extract_year",
                                    "new_column": f"{col_to_transform}_year"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_year' with extracted years")
                        
                        elif transform_method == "Extract Month":
                            self.df[f"{col_to_transform}_month"] = self.df[date_col].dt.month
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Extracted month from '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "extract_month",
                                    "new_column": f"{col_to_transform}_month"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_month' with extracted months")
                        
                        elif transform_method == "Extract Day":
                            self.df[f"{col_to_transform}_day"] = self.df[date_col].dt.day
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Extracted day from '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "extract_day",
                                    "new_column": f"{col_to_transform}_day"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_day' with extracted days")
                        
                        elif transform_method == "Extract Day of Week":
                            self.df[f"{col_to_transform}_dayofweek"] = self.df[date_col].dt.dayofweek
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Extracted day of week from '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "extract_dayofweek",
                                    "new_column": f"{col_to_transform}_dayofweek"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_dayofweek' with extracted days of week (0=Monday, 6=Sunday)")
                        
                        elif transform_method == "Extract Hour":
                            self.df[f"{col_to_transform}_hour"] = self.df[date_col].dt.hour
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Extracted hour from '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "extract_hour",
                                    "new_column": f"{col_to_transform}_hour"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_hour' with extracted hours")
                        
                        elif transform_method == "Extract Season":
                            # Define a function to get season from month
                            def get_season(month):
                                if month in [12, 1, 2]:
                                    return 'Winter'
                                elif month in [3, 4, 5]:
                                    return 'Spring'
                                elif month in [6, 7, 8]:
                                    return 'Summer'
                                else:
                                    return 'Fall'
                            
                            self.df[f"{col_to_transform}_season"] = self.df[date_col].dt.month.apply(get_season)
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Extracted season from '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "extract_season",
                                    "new_column": f"{col_to_transform}_season"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_season' with extracted seasons")
                        
                        elif transform_method == "Days Since Reference Date":
                            ref_date_pd = pd.Timestamp(ref_date)
                            self.df[f"{col_to_transform}_days_since_ref"] = (self.df[date_col] - ref_date_pd).dt.days
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Calculated days since {ref_date} for '{col_to_transform}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "transformation",
                                "details": {
                                    "column": col_to_transform,
                                    "method": "days_since_ref",
                                    "reference_date": str(ref_date),
                                    "new_column": f"{col_to_transform}_days_since_ref"
                                }
                            })
                            
                            st.success(f"Created new column '{col_to_transform}_days_since_ref' with days since {ref_date}")
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying transformation: {str(e)}")
    
    def _render_feature_engineering(self):
        """Render feature engineering interface"""
        st.subheader("Feature Engineering")
        
        # Create columns for organized layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Create New Features")
            
            # Feature creation methods
            feature_method = st.selectbox(
                "Feature creation method:",
                [
                    "Arithmetic Operation",
                    "Mathematical Function",
                    "String Operation",
                    "Conditional Logic",
                    "Aggregation by Group",
                    "Rolling Window"
                ],
                key="feature_method"
            )
            
            # Different options based on the selected method
            if feature_method == "Arithmetic Operation":
                # Get numeric columns
                num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
                
                if len(num_cols) < 1:
                    st.warning("Arithmetic operations require at least one numeric column.")
                else:
                    col1 = st.selectbox("Select first column:", num_cols, key="arith_col1")
                    
                    operation = st.selectbox(
                        "Select operation:",
                        ["+", "-", "*", "/", "^", "%"],
                        key="arith_op"
                    )
                    
                    # Second operand can be a column or a constant
                    use_constant = st.checkbox("Use constant value for second operand", key="use_const")
                    
                    if use_constant:
                        constant = st.number_input("Enter constant value:", value=1.0, key="arith_const")
                    else:
                        col2_options = [col for col in num_cols if col != col1]
                        if not col2_options:
                            st.warning("No second column available. Please use a constant.")
                            col2 = None
                        else:
                            col2 = st.selectbox("Select second column:", col2_options, key="arith_col2")
                    
                    new_col_name = st.text_input("New column name:", key="arith_new_col")
                    
                    if st.button("Create Feature", key="create_arith"):
                        try:
                            if not new_col_name:
                                st.error("Please provide a name for the new column")
                            else:
                                # Apply the operation
                                if operation == "+":
                                    if use_constant:
                                        self.df[new_col_name] = self.df[col1] + constant
                                    else:
                                        self.df[new_col_name] = self.df[col1] + self.df[col2]
                                elif operation == "-":
                                    if use_constant:
                                        self.df[new_col_name] = self.df[col1] - constant
                                    else:
                                        self.df[new_col_name] = self.df[col1] - self.df[col2]
                                elif operation == "*":
                                    if use_constant:
                                        self.df[new_col_name] = self.df[col1] * constant
                                    else:
                                        self.df[new_col_name] = self.df[col1] * self.df[col2]
                                elif operation == "/":
                                    if use_constant:
                                        if constant == 0:
                                            st.error("Cannot divide by zero")
                                            return
                                        self.df[new_col_name] = self.df[col1] / constant
                                    else:
                                        self.df[new_col_name] = self.df[col1] / self.df[col2].replace(0, np.nan)
                                elif operation == "^":
                                    if use_constant:
                                        self.df[new_col_name] = self.df[col1] ** constant
                                    else:
                                        self.df[new_col_name] = self.df[col1] ** self.df[col2]
                                elif operation == "%":
                                    if use_constant:
                                        if constant == 0:
                                            st.error("Cannot take modulo with zero")
                                            return
                                        self.df[new_col_name] = self.df[col1] % constant
                                    else:
                                        self.df[new_col_name] = self.df[col1] % self.df[col2].replace(0, np.nan)
                                
                                # Add to processing history
                                second_operand = str(constant) if use_constant else col2
                                st.session_state.processing_history.append({
                                    "description": f"Created feature '{new_col_name}' using {col1} {operation} {second_operand}",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "feature_engineering",
                                    "details": {
                                        "method": "arithmetic",
                                        "column1": col1,
                                        "operation": operation,
                                        "operand2": second_operand,
                                        "new_column": new_col_name
                                    }
                                })
                                
                                st.success(f"Created new column '{new_col_name}'")
                                
                                # Update the dataframe in session state
                                st.session_state.df = self.df
                                st.experimental_rerun()
                                
                        except Exception as e:
                            st.error(f"Error creating feature: {str(e)}")
            
            elif feature_method == "Mathematical Function":
                # Get numeric columns
                num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
                
                if not num_cols:
                    st.warning("Mathematical functions require at least one numeric column.")
                else:
                    col = st.selectbox("Select column:", num_cols, key="math_col")
                    
                    function = st.selectbox(
                        "Select function:",
                        ["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "round", "ceil", "floor"],
                        key="math_func"
                    )
                    
                    new_col_name = st.text_input("New column name:", key="math_new_col")
                    
                    if st.button("Create Feature", key="create_math"):
                        try:
                            if not new_col_name:
                                st.error("Please provide a name for the new column")
                            else:
                                # Apply the function
                                if function == "sin":
                                    self.df[new_col_name] = np.sin(self.df[col])
                                elif function == "cos":
                                    self.df[new_col_name] = np.cos(self.df[col])
                                elif function == "tan":
                                    self.df[new_col_name] = np.tan(self.df[col])
                                elif function == "exp":
                                    self.df[new_col_name] = np.exp(self.df[col])
                                elif function == "log":
                                    # Handle non-positive values
                                    self.df[new_col_name] = np.log(self.df[col].clip(lower=1e-10))
                                elif function == "sqrt":
                                    # Handle negative values
                                    self.df[new_col_name] = np.sqrt(self.df[col].clip(lower=0))
                                elif function == "abs":
                                    self.df[new_col_name] = np.abs(self.df[col])
                                elif function == "round":
                                    self.df[new_col_name] = np.round(self.df[col])
                                elif function == "ceil":
                                    self.df[new_col_name] = np.ceil(self.df[col])
                                elif function == "floor":
                                    self.df[new_col_name] = np.floor(self.df[col])
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Created feature '{new_col_name}' using {function}({col})",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "feature_engineering",
                                    "details": {
                                        "method": "mathematical_function",
                                        "column": col,
                                        "function": function,
                                        "new_column": new_col_name
                                    }
                                })
                                
                                st.success(f"Created new column '{new_col_name}'")
                                
                                # Update the dataframe in session state
                                st.session_state.df = self.df
                                st.experimental_rerun()
                                
                        except Exception as e:
                            st.error(f"Error creating feature: {str(e)}")
            
            elif feature_method == "String Operation":
                # Get string (object) columns
                str_cols = self.df.select_dtypes(include=['object']).columns.tolist()
                
                if not str_cols:
                    st.warning("String operations require at least one string column.")
                else:
                    col = st.selectbox("Select column:", str_cols, key="str_col")
                    
                    operation = st.selectbox(
                        "Select operation:",
                        [
                            "To Uppercase",
                            "To Lowercase",
                            "Extract Substring",
                            "String Length",
                            "Replace Text",
                            "Remove Whitespace",
                            "Extract Pattern (Regex)"
                        ],
                        key="str_op"
                    )
                    
                    # Additional parameters for specific operations
                    if operation == "Extract Substring":
                        start_idx = st.number_input("Start index:", value=0, min_value=0, key="substr_start")
                        end_idx = st.number_input("End index:", value=5, min_value=0, key="substr_end")
                    elif operation == "Replace Text":
                        old_text = st.text_input("Text to replace:", key="replace_old")
                        new_text = st.text_input("Replacement text:", key="replace_new")
                    elif operation == "Extract Pattern (Regex)":
                        pattern = st.text_input("Regex pattern:", key="regex_pattern")
                    
                    new_col_name = st.text_input("New column name:", key="str_new_col")
                    
                    if st.button("Create Feature", key="create_str"):
                        try:
                            if not new_col_name:
                                st.error("Please provide a name for the new column")
                            else:
                                # Apply the operation
                                if operation == "To Uppercase":
                                    self.df[new_col_name] = self.df[col].str.upper()
                                    operation_desc = "uppercase"
                import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import datetime
import re

class DataProcessor:
    """Class for processing and transforming data"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
        self.original_df = df.copy() if df is not None else None
        
        # Store processing history
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
    
    def render_interface(self):
        """Render the data processing interface"""
        st.header("Data Processing")
        
        if self.df is None or self.df.empty:
            st.info("Please upload a dataset to begin data processing.")
            return
        
        # Create tabs for different processing tasks
        processing_tabs = st.tabs([
            "Data Cleaning",
            "Data Transformation",
            "Feature Engineering",
            "Data Filtering",
            "Column Management"
        ])
        
        # Data Cleaning Tab
        with processing_tabs[0]:
            self._render_data_cleaning()
        
        # Data Transformation Tab
        with processing_tabs[1]:
            self._render_data_transformation()
        
        # Feature Engineering Tab
        with processing_tabs[2]:
            self._render_feature_engineering()
        
        # Data Filtering Tab
        with processing_tabs[3]:
            self._render_data_filtering()
        
        # Column Management Tab
        with processing_tabs[4]:
            self._render_column_management()
        
        # Processing History
        if st.session_state.processing_history:
            st.header("Processing History")
            
            # Create collapsible section for history
            with st.expander("View Processing Steps", expanded=False):
                for i, step in enumerate(st.session_state.processing_history):
                    st.markdown(f"**Step {i+1}:** {step['description']} - {step['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Reset button
            if st.button("Reset to Original Data", key="reset_data"):
                self.df = self.original_df.copy()
                st.session_state.df = self.original_df.copy()
                st.session_state.processing_history = []
                st.success("Data reset to original state!")
                st.experimental_rerun()
    
    def _render_data_cleaning(self):
        """Render data cleaning interface"""
        st.subheader("Data Cleaning")
        
        # Create columns for organized layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Missing Values")
            
            # Show missing value statistics
            missing_vals = self.df.isna().sum()
            total_missing = missing_vals.sum()
            
            if total_missing == 0:
                st.success("No missing values found in the dataset!")
            else:
                st.warning(f"Found {total_missing} missing values across {(missing_vals > 0).sum()} columns")
                
                # Display columns with missing values
                cols_with_missing = self.df.columns[missing_vals > 0].tolist()
                
                # Create a dataframe to show missing value statistics
                missing_df = pd.DataFrame({
                    'Column': cols_with_missing,
                    'Missing Values': [missing_vals[col] for col in cols_with_missing],
                    'Percentage': [missing_vals[col] / len(self.df) * 100 for col in cols_with_missing]
                })
                
                st.dataframe(missing_df, use_container_width=True)
                
                # Options for handling missing values
                st.markdown("#### Handle Missing Values")
                
                col_to_handle = st.selectbox(
                    "Select column to handle:",
                    cols_with_missing
                )
                
                handling_method = st.selectbox(
                    "Select handling method:",
                    [
                        "Drop rows",
                        "Fill with mean",
                        "Fill with median",
                        "Fill with mode",
                        "Fill with constant value",
                        "Fill with forward fill",
                        "Fill with backward fill"
                    ]
                )
                
                # Additional input for constant value if selected
                if handling_method == "Fill with constant value":
                    constant_value = st.text_input("Enter constant value:")
                
                # Apply button
                if st.button("Apply Missing Value Treatment", key="apply_missing"):
                    try:
                        # Store original shape for reporting
                        orig_shape = self.df.shape
                        
                        # Apply the selected method
                        if handling_method == "Drop rows":
                            self.df = self.df.dropna(subset=[col_to_handle])
                            st.session_state.df = self.df
                            
                            # Add to processing history
                            rows_removed = orig_shape[0] - self.df.shape[0]
                            st.session_state.processing_history.append({
                                "description": f"Dropped {rows_removed} rows with missing values in column '{col_to_handle}'",
                                "timestamp": datetime.datetime.now(),
                                "type": "missing_values",
                                "details": {
                                    "column": col_to_handle,
                                    "method": "drop",
                                    "rows_affected": rows_removed
                                }
                            })
                            
                            st.success(f"Dropped {rows_removed} rows with missing values in '{col_to_handle}'")
                            
                        elif handling_method == "Fill with mean":
                            if self.df[col_to_handle].dtype.kind in 'bifc':  # Check if numeric
                                mean_val = self.df[col_to_handle].mean()
                                self.df[col_to_handle] = self.df[col_to_handle].fillna(mean_val)
                                st.session_state.df = self.df
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Filled missing values in '{col_to_handle}' with mean ({mean_val:.2f})",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "missing_values",
                                    "details": {
                                        "column": col_to_handle,
                                        "method": "mean",
                                        "value": mean_val
                                    }
                                })
                                
                                st.success(f"Filled missing values in '{col_to_handle}' with mean: {mean_val:.2f}")
                            else:
                                st.error(f"Column '{col_to_handle}' is not numeric. Cannot use mean imputation.")
                                
                        elif handling_method == "Fill with median":
                            if self.df[col_to_handle].dtype.kind in 'bifc':  # Check if numeric
                                median_val = self.df[col_to_handle].median()
                                self.df[col_to_handle] = self.df[col_to_handle].fillna(median_val)
                                st.session_state.df = self.df
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Filled missing values in '{col_to_handle}' with median ({median_val:.2f})",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "missing_values",
                                    "details": {
                                        "column": col_to_handle,
                                        "method": "median",
                                        "value": median_val
                                    }
                                })
                                
                                st.success(f"Filled missing values in '{col_to_handle}' with median: {median_val:.2f}")
                            else:
                                st.error(f"Column '{col_to_handle}' is not numeric. Cannot use median imputation.")
                                
                        elif handling_method == "Fill with mode":
                            mode_val = self.df[col_to_handle].mode()[0]
                            self.df[col_to_handle] = self.df[col_to_handle].fillna(mode_val)
                            st.session_state.df = self.df
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Filled missing values in '{col_to_handle}' with mode ({mode_val})",
                                "timestamp": datetime.datetime.now(),
                                "type": "missing_values",
                                "details": {
                                    "column": col_to_handle,
                                    "method": "mode",
                                    "value": mode_val
                                }
                            })
                            
                            st.success(f"Filled missing values in '{col_to_handle}' with mode: {mode_val}")
                            
                        elif handling_method == "Fill with constant value":
                            if constant_value:
                                # Try to convert to appropriate type
                                try:
                                    if self.df[col_to_handle].dtype.kind in 'bifc':  # Numeric
                                        constant_val = float(constant_value)
                                    elif self.df[col_to_handle].dtype.kind == 'b':  # Boolean
                                        constant_val = constant_value.lower() in ['true', 'yes', '1', 't', 'y']
                                    else:
                                        constant_val = constant_value
                                        
                                    self.df[col_to_handle] = self.df[col_to_handle].fillna(constant_val)
                                    st.session_state.df = self.df
                                    
                                    # Add to processing history
                                    st.session_state.processing_history.append({
                                        "description": f"Filled missing values in '{col_to_handle}' with constant ({constant_val})",
                                        "timestamp": datetime.datetime.now(),
                                        "type": "missing_values",
                                        "details": {
                                            "column": col_to_handle,
                                            "method": "constant",
                                            "value": constant_val
                                        }
                                    })
                                    
                                    st.success(f"Filled missing values in '{col_to_handle}' with: {constant_val}")
                                except ValueError:
                                    st.error(f"Could not convert '{constant_value}' to appropriate type for column '{col_to_handle}'")
                            else:
                                st.error("Please enter a constant value")
                                
                        elif handling_method == "Fill with forward fill":
                            self.df[col_to_handle] = self.df[col_to_handle].fillna(method='ffill')
                            st.session_state.df = self.df
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Filled missing values in '{col_to_handle}' with forward fill",
                                "timestamp": datetime.datetime.now(),
                                "type": "missing_values",
                                "details": {
                                    "column": col_to_handle,
                                    "method": "ffill"
                                }
                            })
                            
                            st.success(f"Filled missing values in '{col_to_handle}' with forward fill")
                            
                        elif handling_method == "Fill with backward fill":
                            self.df[col_to_handle] = self.df[col_to_handle].fillna(method='bfill')
                            st.session_state.df = self.df
                            
                            # Add to processing history
                            st.session_state.processing_history.append({
                                "description": f"Filled missing values in '{col_to_handle}' with backward fill",
                                "timestamp": datetime.datetime.now(),
                                "type": "missing_values",
                                "details": {
                                    "column": col_to_handle,
                                    "method": "bfill"
                                }
                            })
                            
                            st.success(f"Filled missing values in '{col_to_handle}' with backward fill")
                    
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        st.experimental_rerun()
                            
                    except Exception as e:
                        st.error(f"Error handling missing values: {str(e)}")
        
        with col2:
            st.markdown("### Duplicate Rows")
            
            # Check for duplicates
            dup_count = self.df.duplicated().sum()
            
            if dup_count == 0:
                st.success("No duplicate rows found in the dataset!")
            else:
                st.warning(f"Found {dup_count} duplicate rows in the dataset")
                
                # Display sample of duplicates
                if st.checkbox("Show sample of duplicates"):
                    duplicates = self.df[self.df.duplicated(keep='first')]
                    st.dataframe(duplicates.head(5), use_container_width=True)
                
                # Button to remove duplicates
                if st.button("Remove Duplicate Rows"):
                    try:
                        # Store original shape for reporting
                        orig_shape = self.df.shape
                        
                        # Remove duplicates
                        self.df = self.df.drop_duplicates()
                        
                        # Update the dataframe in session state
                        st.session_state.df = self.df
                        
                        # Add to processing history
                        rows_removed = orig_shape[0] - self.df.shape[0]
                        st.session_state.processing_history.append({
                            "description": f"Removed {rows_removed} duplicate rows",
                            "timestamp": datetime.datetime.now(),
                            "type": "duplicates",
                            "details": {
                                "rows_removed": rows_removed
                            }
                        })
                        
                        st.success(f"Removed {rows_removed} duplicate rows")
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error removing duplicates: {str(e)}")
            
            # Outlier Detection and Handling
            st.markdown("### Outlier Detection")
            
            # Get numeric columns for outlier detection
            num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if not num_cols:
                st.info("No numeric columns found for outlier detection.")
            else:
                col_for_outliers = st.selectbox(
                    "Select column for outlier detection:",
                    num_cols
                )
                
                # Calculate outlier bounds using IQR method
                Q1 = self.df[col_for_outliers].quantile(0.25)
                Q3 = self.df[col_for_outliers].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers = self.df[(self.df[col_for_outliers] < lower_bound) | (self.df[col_for_outliers] > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count == 0:
                    st.success(f"No outliers found in column '{col_for_outliers}'")
                else:
                    st.warning(f"Found {outlier_count} outliers in column '{col_for_outliers}'")
                    st.write(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                    
                    # Display sample of outliers
                    if st.checkbox("Show sample of outliers"):
                        st.dataframe(outliers.head(5), use_container_width=True)
                    
                    # Options for handling outliers
                    outlier_method = st.selectbox(
                        "Select handling method:",
                        [
                            "Remove outliers",
                            "Cap outliers (winsorize)",
                            "Replace with NaN"
                        ]
                    )
                    
                    # Button to handle outliers
                    if st.button("Handle Outliers"):
                        try:
                            # Store original shape for reporting
                            orig_shape = self.df.shape
                            
                            if outlier_method == "Remove outliers":
                                # Remove rows with outliers
                                self.df = self.df[
                                    (self.df[col_for_outliers] >= lower_bound) & 
                                    (self.df[col_for_outliers] <= upper_bound)
                                ]
                                
                                # Add to processing history
                                rows_removed = orig_shape[0] - self.df.shape[0]
                                st.session_state.processing_history.append({
                                    "description": f"Removed {rows_removed} outliers from column '{col_for_outliers}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "outliers",
                                    "details": {
                                        "column": col_for_outliers,
                                        "method": "remove",
                                        "rows_affected": rows_removed
                                    }
                                })
                                
                                st.success(f"Removed {rows_removed} outliers from column '{col_for_outliers}'")
                                
                            elif outlier_method == "Cap outliers (winsorize)":
                                # Cap values at the bounds
                                self.df[col_for_outliers] = self.df[col_for_outliers].clip(lower=lower_bound, upper=upper_bound)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Capped {outlier_count} outliers in column '{col_for_outliers}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "outliers",
                                    "details": {
                                        "column": col_for_outliers,
                                        "method": "cap",
                                        "lower_bound": lower_bound,
                                        "upper_bound": upper_bound,
                                        "values_affected": outlier_count
                                    }
                                })
                                
                                st.success(f"Capped {outlier_count} outliers in column '{col_for_outliers}'")
                                
                            elif outlier_method == "Replace with NaN":
                                # Replace outliers with NaN
                                self.df.loc[
                                    (self.df[col_for_outliers] < lower_bound) | 
                                    (self.df[col_for_outliers] > upper_bound),
                                    col_for_outliers
                                ] = np.nan
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Replaced {outlier_count} outliers with NaN in column '{col_for_outliers}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "outliers",
                                    "details": {
                                        "column": col_for_outliers,
                                        "method": "replace_nan",
                                        "values_affected": outlier_count
                                    }
                                })
                                
                                st.success(f"Replaced {outlier_count} outliers with NaN in column '{col_for_outliers}'")
                            
                            # Update the dataframe in session state
                            st.session_state.df = self.df
                            st.experimental_rerun()
                            
                        except Exception as e:
                            st.error(f"Error handling outliers: {str(e)}")
    
    def _render_data_transformation(self):
        """Render data transformation interface"""
        st.subheader("Data Transformation")
        
        # Create columns for organized layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Numeric Transformations")
            
            # Get numeric columns
            num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            if not num_cols:
                st.info("No numeric columns found for transformation.")
            else:
                col_to_transform = st.selectbox(
                    "Select column to transform:",
                    num_cols,
                    key="num_transform_col"
                )
                
                transform_method = st.selectbox(
                    "Select transformation method:",
                    [
                        "Standardization (Z-score)",
                        "Min-Max Scaling",
                        "Log Transform",
                        "Square Root Transform",
                        "Box-Cox Transform",
                        "Binning/Discretization"
                    ],
                    key="transform_method"
                )
                
                # Additional parameters for specific transforms
                if transform_method == "Binning/Discretization":
                    num_bins = st.slider("Number of bins:", 2, 20, 5)
                    bin_labels = st.text_input("Bin labels (comma-separated, leave empty for default):")
                
                # Apply button
                if st.button("Apply Transformation", key="apply_transform"):
                    try:
                        if transform_method == "Standardization (Z-score)":
                            # Z-score standardization: (x - mean) / std
                            mean = self.df[col_to_transform].mean()
                            std = self.df[col_to_transform].std()
                            
                            if std == 0:
                                st.error(f"Standard deviation is zero for column '{col_to_transform}'. Cannot standardize.")
                            else:
                                self.df[f"{col_to_transform}_standardized"] = (self.df[col_to_transform] - mean) / std
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied standardization to '{col_to_transform}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "standardization",
                                        "new_column": f"{col_to_transform}_standardized"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_standardized' with standardized values")
                        
                        elif transform_method == "Min-Max Scaling":
                            # Min-Max scaling: (x - min) / (max - min)
                            min_val = self.df[col_to_transform].min()
                            max_val = self.df[col_to_transform].max()
                            
                            if max_val == min_val:
                                st.error(f"Maximum and minimum values are the same for column '{col_to_transform}'. Cannot scale.")
                            else:
                                self.df[f"{col_to_transform}_scaled"] = (self.df[col_to_transform] - min_val) / (max_val - min_val)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied min-max scaling to '{col_to_transform}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "min_max_scaling",
                                        "new_column": f"{col_to_transform}_scaled"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_scaled' with scaled values (0-1)")
                        
                        elif transform_method == "Log Transform":
                            # Check for non-positive values
                            if (self.df[col_to_transform] <= 0).any():
                                min_val = self.df[col_to_transform].min()
                                # Add a constant to make all values positive
                                const = abs(min_val) + 1 if min_val <= 0 else 0
                                self.df[f"{col_to_transform}_log"] = np.log(self.df[col_to_transform] + const)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied log transform to '{col_to_transform}' with constant {const}",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "log_transform",
                                        "constant": const,
                                        "new_column": f"{col_to_transform}_log"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_log' with log transform (added constant {const})")
                            else:
                                self.df[f"{col_to_transform}_log"] = np.log(self.df[col_to_transform])
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied log transform to '{col_to_transform}'",
                                    "timestamp": datetime.datetime.now(),
                                    "type": "transformation",
                                    "details": {
                                        "column": col_to_transform,
                                        "method": "log_transform",
                                        "new_column": f"{col_to_transform}_log"
                                    }
                                })
                                
                                st.success(f"Created new column '{col_to_transform}_log' with log transform")
                        
                        elif transform_method == "Square Root Transform":
                            # Check for negative values
                            if (self.df[col_to_transform] < 0).any():
                                min_val = self.df[col_to_transform].min()
                                # Add a constant to make all values non-negative
                                const = abs(min_val) + 1 if min_val < 0 else 0
                                self.df[f"{col_to_transform}_sqrt"] = np.sqrt(self.df[col_to_transform] + const)
                                
                                # Add to processing history
                                st.session_state.processing_history.append({
                                    "description": f"Applied square root transform