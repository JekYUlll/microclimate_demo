# 微气候预测项目架构图

## 1. 系统整体架构图

```mermaid
graph TB
    subgraph "数据源"
        DB[(MySQL数据库<br/>南极气象站数据)]
    end
    
    subgraph "训练阶段"
        TRAIN[模型训练<br/>train.ipynb]
        MODEL[TFT模型<br/>tft_model.pt]
        SCALER[数据缩放器<br/>scaler.pkl]
    end
    
    subgraph "预测服务"
        API[FastAPI服务<br/>app.py]
        WS[WebSocket<br/>实时推送]
        WEB[Web界面<br/>static/index.html]
    end
    
    subgraph "数据模拟"
        MOCK[Go模拟器<br/>main.go]
    end
    
    DB --> TRAIN
    TRAIN --> MODEL
    TRAIN --> SCALER
    
    DB --> MOCK
    MOCK --> API
    
    MODEL --> API
    SCALER --> API
    
    API --> WS
    API --> WEB
    
    style DB fill:#e1f5fe
    style TRAIN fill:#f3e5f5
    style API fill:#e8f5e8
    style MOCK fill:#fff3e0
```

## 2. 数据流程图

```mermaid
flowchart TD
    subgraph "数据采集"
        A[南极气象站] --> B[MySQL数据库]
    end
    
    subgraph "模型训练"
        B --> C[数据预处理]
        C --> D[时间序列构建]
        D --> E[TFT模型训练]
        E --> F[模型保存]
    end
    
    subgraph "实时预测"
        G[Go模拟器] --> H[数据注入]
        H --> I[FastAPI接收]
        I --> J[时间序列缓存]
        J --> K[模型预测]
        K --> L[WebSocket推送]
        L --> M[Web界面展示]
    end
    
    F --> K
    
    style A fill:#ffebee
    style E fill:#e8f5e8
    style K fill:#e3f2fd
```

## 3. 技术栈图

```mermaid
graph TD
    subgraph "机器学习"
        DARTS[Darts库<br/>时间序列预测]
        TFT[TFT模型<br/>Temporal Fusion Transformer]
        PYTORCH[PyTorch<br/>深度学习框架]
    end
    
    subgraph "后端服务"
        FASTAPI[FastAPI<br/>Web框架]
        WEBSOCKET[WebSocket<br/>实时通信]
        PANDAS[Pandas<br/>数据处理]
    end
    
    subgraph "数据存储"
        MYSQL[MySQL<br/>历史数据]
        CACHE[内存缓存<br/>实时数据]
    end
    
    subgraph "前端展示"
        HTML[HTML/CSS/JS<br/>Web界面]
    end
    
    subgraph "数据模拟"
        GO[Go语言<br/>数据模拟器]
    end
    
    DARTS --> TFT
    TFT --> PYTORCH
    FASTAPI --> WEBSOCKET
    FASTAPI --> PANDAS
    MYSQL --> CACHE
    WEBSOCKET --> HTML
    GO --> FASTAPI
    
    style DARTS fill:#e8f5e8
    style FASTAPI fill:#e3f2fd
    style MYSQL fill:#fff3e0
    style GO fill:#fce4ec
```

## 4. 模型训练流程图

```mermaid
flowchart TD
    A[读取MySQL数据] --> B[数据预处理]
    B --> C[选择训练站点]
    C --> D[时间序列重采样<br/>5分钟频率]
    D --> E[数据插值填充]
    E --> F[创建Darts TimeSeries]
    F --> G[数据标准化]
    G --> H[划分训练/验证集]
    H --> I[配置TFT模型参数]
    I --> J[模型训练]
    J --> K[验证集预测]
    K --> L[计算误差指标]
    L --> M[保存模型和缩放器]
    
    style A fill:#e1f5fe
    style J fill:#e8f5e8
    style M fill:#f3e5f5
```

## 5. 实时预测流程图

```mermaid
sequenceDiagram
    participant M as Go模拟器
    participant A as FastAPI
    participant C as 数据缓存
    participant T as TFT模型
    participant W as WebSocket
    participant U as 用户界面
    
    M->>A: POST /ingest
    A->>C: 更新站点数据
    C->>A: 检查数据长度
    alt 数据足够(≥24点)
        A->>T: 构建时间序列
        T->>A: 返回预测结果
        A->>W: 广播预测更新
        W->>U: 实时推送数据
    else 数据不足
        A->>M: 返回状态信息
    end
```

## 6. 项目文件结构图

```mermaid
graph TD
    A[microclimate_demo/] --> B[app.py<br/>FastAPI服务]
    A --> C[train.ipynb<br/>模型训练]
    A --> D[static/index.html<br/>Web界面]
    A --> E[go/sensor_moc/main.go<br/>数据模拟器]
    A --> F[tft_model.pt<br/>训练好的模型]
    A --> G[scaler.pkl<br/>数据缩放器]
    A --> H[training_info.pkl<br/>训练信息]
    A --> I[log/<br/>日志文件]
    
    style B fill:#e8f5e8
    style C fill:#f3e5f5
    style E fill:#fff3e0
    style F fill:#e1f5fe
```

## 7. 数据特征图

```mermaid
graph LR
    subgraph "输入特征"
        T[温度 Temperature]
        H[湿度 Humidity]
        W[风速 Wind Speed]
    end
    
    subgraph "时间窗口"
        I[输入窗口<br/>24个时间点<br/>2小时历史]
        O[输出窗口<br/>6个时间点<br/>30分钟预测]
    end
    
    subgraph "预测结果"
        PT[预测温度]
        PH[预测湿度]
        PW[预测风速]
    end
    
    T --> I
    H --> I
    W --> I
    I --> O
    O --> PT
    O --> PH
    O --> PW
    
    style I fill:#e3f2fd
    style O fill:#e8f5e8
    style PT fill:#fff3e0
    style PH fill:#fff3e0
    style PW fill:#fff3e0
```

## 项目特点总结

1. **完整的数据管道**: 从历史数据训练到实时预测
2. **先进的时间序列模型**: 使用TFT (Temporal Fusion Transformer)
3. **实时预测服务**: FastAPI + WebSocket实现实时推送
4. **多站点支持**: 支持多个南极气象站的数据
5. **数据模拟工具**: Go语言编写的传感器数据模拟器
6. **Web界面展示**: 直观的数据可视化界面
