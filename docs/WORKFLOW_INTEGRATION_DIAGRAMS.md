# WorkflowAgent Integration - Visual Flow

## Complete Pipeline Flow

```mermaid
graph TD
    A[User Request] --> B[Intent Agent]
    B --> C[Discovery Agent]
    C --> D[Modeling Agent]
    D --> E[Transformation Agent]
    E --> F{Task Type?}
    
    F -->|ML| G[ML Agent]
    F -->|Analytics| H[Quality Agent]
    
    G --> I[Packaging Agent]
    H --> I
    
    I --> J[Workflow Agent ⭐ NEW]
    
    J --> K[Airflow DAG]
    J --> L[Cron Job]
    
    K --> M[Deployable Workflows]
    L --> M
    
    style J fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style K fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
    style L fill:#FF9800,stroke:#E65100,stroke-width:2px,color:#fff
    style M fill:#9C27B0,stroke:#6A1B9A,stroke-width:2px,color:#fff
```

## State Flow

```mermaid
stateDiagram-v2
    [*] --> user_request
    user_request --> intent
    intent --> discovery_result
    discovery_result --> data_model
    data_model --> transformation
    transformation --> quality_checks
    transformation --> ml_result
    quality_checks --> data_product_spec
    ml_result --> data_product_spec
    data_product_spec --> yaml_output
    yaml_output --> workflow_result
    workflow_result --> [*]
    
    note right of workflow_result
        NEW: Workflow Generation
        - dag_code
        - cron_code
        - schedule
        - file paths
    end note
```

## Workflow Agent Process

```mermaid
flowchart LR
    A[Data Product Spec] --> B{Workflow Agent}
    
    B --> C[Extract Metadata]
    C --> D[Generate DAG ID]
    
    B --> E[Extract SLA]
    E --> F[Parse Schedule]
    
    B --> G[Extract Transformation]
    G --> H[Prepare Context]
    
    D --> I[Render DAG Template]
    F --> I
    H --> I
    
    D --> J[Render Cron Template]
    F --> J
    H --> J
    
    I --> K[Write DAG File]
    J --> L[Write Cron File]
    
    K --> M[workflow_result]
    L --> M
    
    style B fill:#4CAF50,stroke:#2E7D32,stroke-width:3px
    style I fill:#2196F3,stroke:#1565C0,stroke-width:2px
    style J fill:#FF9800,stroke:#E65100,stroke-width:2px
    style M fill:#9C27B0,stroke:#6A1B9A,stroke-width:2px
```

## Generated DAG Structure

```mermaid
graph LR
    A[validate_source_freshness] --> B[execute_transformation]
    B --> C[run_quality_checks]
    C --> D[update_registry_metadata]
    
    style A fill:#FFC107,stroke:#F57F17,stroke-width:2px
    style B fill:#2196F3,stroke:#1565C0,stroke-width:2px
    style C fill:#4CAF50,stroke:#2E7D32,stroke-width:2px
    style D fill:#9C27B0,stroke:#6A1B9A,stroke-width:2px
```

## Integration Points

```mermaid
graph TB
    subgraph "Orchestrator Pipeline"
        A[Intent] --> B[Discovery]
        B --> C[Modeling]
        C --> D[Transformation]
        D --> E[Quality]
        E --> F[Packaging]
        F --> G[Workflow ⭐]
    end
    
    subgraph "Workflow Agent"
        G --> H[Template Engine]
        H --> I[Jinja2 Renderer]
        I --> J[File Writer]
    end
    
    subgraph "Output"
        J --> K[Airflow DAG]
        J --> L[Cron Script]
    end
    
    subgraph "Deployment"
        K --> M[Airflow Scheduler]
        L --> N[Crontab]
    end
    
    style G fill:#4CAF50,stroke:#2E7D32,stroke-width:3px
    style H fill:#2196F3,stroke:#1565C0,stroke-width:2px
    style I fill:#FF9800,stroke:#E65100,stroke-width:2px
    style J fill:#9C27B0,stroke:#6A1B9A,stroke-width:2px
```

## Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant O as Orchestrator
    participant P as Packaging Agent
    participant W as Workflow Agent
    participant F as File System
    
    U->>O: Natural Language Request
    O->>O: Process through agents
    O->>P: Package specification
    P-->>O: data_product_spec
    O->>W: Generate workflows
    W->>W: Extract metadata & SLA
    W->>W: Render DAG template
    W->>F: Write DAG file
    W->>W: Render cron template
    W->>F: Write cron file
    W-->>O: workflow_result
    O-->>U: Complete result
    
    Note over W,F: NEW: Workflow Generation
```

## Component Architecture

```mermaid
graph TB
    subgraph "Core Components"
        A[OrchestratorAgent]
        B[WorkflowAgent]
        C[PackagingAgent]
    end
    
    subgraph "Templates"
        D[airflow_dag.py.j2]
        E[cron_job.sh.j2]
    end
    
    subgraph "Dependencies"
        F[Jinja2]
        G[croniter]
    end
    
    subgraph "Output"
        H[generated_workflows/dags/]
        I[generated_workflows/cron/]
    end
    
    A --> B
    A --> C
    C --> B
    B --> F
    B --> G
    B --> D
    B --> E
    D --> H
    E --> I
    
    style B fill:#4CAF50,stroke:#2E7D32,stroke-width:3px
```

## Schedule Extraction Logic

```mermaid
flowchart TD
    A[SLA Freshness] --> B{Contains 'hourly'?}
    B -->|Yes| C[@hourly]
    B -->|No| D{Contains 'daily'?}
    D -->|Yes| E{Time specified?}
    E -->|Yes| F[Extract HH:MM]
    E -->|No| G[Default 6 AM]
    F --> H[MM HH * * *]
    G --> H
    D -->|No| I{Contains 'weekly'?}
    I -->|Yes| J{Day specified?}
    J -->|Yes| K[0 6 * * DAY]
    J -->|No| L[0 6 * * 1]
    I -->|No| M{Contains 'monthly'?}
    M -->|Yes| N[0 6 1 * *]
    M -->|No| O[Default: 0 6 * * *]
    
    style H fill:#4CAF50,stroke:#2E7D32,stroke-width:2px
    style K fill:#4CAF50,stroke:#2E7D32,stroke-width:2px
    style L fill:#4CAF50,stroke:#2E7D32,stroke-width:2px
    style N fill:#4CAF50,stroke:#2E7D32,stroke-width:2px
    style O fill:#4CAF50,stroke:#2E7D32,stroke-width:2px
```
