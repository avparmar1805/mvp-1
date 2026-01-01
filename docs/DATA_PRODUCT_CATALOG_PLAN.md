# Product Plan: Data Product Catalog Dashboard

## 1. Overview
This document outlines the design and implementation plan for a **Central Data Product Catalog**.
**Problem**: Currently, users generate products one-by-one but have no single view to see all the assets they have built.
**Solution**: A "Home" dashboard in the UI that acts as a Registry/Inventory of all generated YAML specifications.

## 2. User Experience (UX)

### 2.1 The "Catalog" View (Home Page)
The default page of the application will act as a collection manager.
*   **Layout**: A Grid or List of "Data Product Cards".
*   **Card Content**:
    *   **Title**: (e.g., "Daily Revenue Report")
    *   **Description**: (e.g., "Daily aggregation of orders...")
    *   **Metadata**: Owner, Created Date, Last Run Status (Success/Fail).
    *   **Tags**: `#sales`, `#finance`, `#marketing`.

### 2.2 Feature Set
1.  **Search & Filter**: Find products by name, domain, or owner.
2.  **Quick Actions**:
    *   **▶️ Run Now**: Trigger an immediate refresh of the data.
    *   **👁️ View**: Open the "Data Preview" and "Visualizations" for this product.
    *   **✏️ Edit**: Load the natural language request into the prompt box for refinement.
    *   **🗑️ Delete**: Remove the valid spec file.

## 3. Technical Architecture

### 3.1 Backend: The "Registry"
For the MVP, we continue using the file system as the database.
*   **Storage**: `output/specs/*.yaml`.
*   **Reader**: A `CatalogService` that scans this directory and parses the YAML headers (`metadata` section).

### 3.2 Frontend: Streamlit Layout
Refactor `ui/app.py` to support multi-page navigation or a "Home" state.

**Proposed Sidebar Navigation**:
*   🏠 **Catalog** (New Default)
*   ➕ **New Data Product** (The current Generator UI)
*   ⚙️ **Settings**

### 3.3 Data Flow
1.  **Load**: `app.py` -> `CatalogService.list_products()` -> Returns `List[DataProductMetadata]`.
2.  **Render**: Loop through list -> `st.card` (or custom container).
3.  **Interact**: Clicking a card sets `session_state.selected_product = product_id` and redirects to the "View" page.

## 4. Future Extensions (Post-MVP)
*   **Lineage View**: Show how Product A relates to Product B in a graph.
*   **Data Marketplace**: allow other users to "Subscribe" to a data product (access control).
*   **Quality Scorecard**: Show a "Health Badges" (e.g., "99% Quality", "Fresh") on the card.
