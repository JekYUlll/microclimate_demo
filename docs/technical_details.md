# 微气候预测项目技术详解

## 1. 核心算法架构图

```mermaid
graph TB
    subgraph "TFT模型内部结构"
        A[输入层<br/>24个时间步] --> B[变量选择网络<br/>Variable Selection]
        B --> C[LSTM编码器<br/>历史信息提取]
        C --> D[LSTM解码器<br/>未来信息生成]
        D --> E[多头注意力机制<br/>Multi-Head Attention]
        E --> F[门控残差网络<br/>Gated Residual Network]
        F --> G[输出层<br/>6个时间步预测]
    end
    
    subgraph "特征处理"
        H[温度] --> I[数据标准化]
        J[湿度] --> I
        K[风速] --> I
        I --> A
    end
    
    style A fill:#e3f2fd
    style G fill:#e8f5e8
    style I fill:#fff3e0
```

## 2. 数据预处理流程图

```mermaid
flowchart TD
    A[原始MySQL数据<br/>15,551条记录] --> B[按站点分组]
    B --> C[选择长城站<br/>4,961条记录]
    C --> D[时间排序]
    D --> E[重采样到5分钟频率<br/>53,088个数据点]
    E --> F[线性插值填充缺失值]
    F --> G[数据标准化<br/>Z-score归一化]
    G --> H[创建Darts TimeSeries]
    H --> I[训练/验证集分割<br/>51,072/2,016]
    
    style A fill:#e1f5fe
    style E fill:#fff3e0
    style I fill:#e8f5e8
```

## 3. 模型性能指标

```mermaid
graph LR
    subgraph "预测精度"
        A[温度 MAE: 0.1059<br/>MSE: 0.0218]
        B[湿度 MAE: 0.2025<br/>MSE: 0.0711]
        C[风速 MAE: 0.2794<br/>MSE: 0.0997]
    end
    
    subgraph "模型参数"
        D[输入窗口: 24步<br/>2小时历史]
        E[输出窗口: 6步<br/>30分钟预测]
        F[隐藏层大小: 32]
        G[训练轮数: 40]
        H[Dropout: 0.1]
    end
    
    style A fill:#e8f5e8
    style B fill:#e8f5e8
    style C fill:#e8f5e8
```

## 4. API接口设计图

```mermaid
graph TB
    subgraph "REST API"
        A[POST /ingest<br/>数据注入]
        B[GET /api/status<br/>服务状态]
        C[GET /predictions/{station}<br/>获取预测]
        D[GET /stations<br/>站点信息]
        E[GET /<br/>Web界面]
    end
    
    subgraph "WebSocket"
        F[/ws<br/>实时推送]
    end
    
    subgraph "数据格式"
        G[WeatherData<br/>station, record_time<br/>temperature, humidity<br/>wind_speed]
    end
    
    A --> G
    F --> G
    
    style A fill:#e3f2fd
    style F fill:#e8f5e8
    style G fill:#fff3e0
```

## 5. 实时数据流图

```mermaid
sequenceDiagram
    participant S as 传感器模拟器
    participant A as FastAPI服务
    participant M as 内存缓存
    participant T as TFT模型
    participant W as WebSocket
    participant C as 客户端
    
    Note over S,C: 数据注入流程
    S->>A: POST /ingest (气象数据)
    A->>M: 更新站点缓存
    M->>A: 返回缓存状态
    
    Note over A,T: 预测流程
    A->>A: 检查数据长度≥24
    A->>T: 构建时间序列
    T->>A: 返回6步预测
    A->>W: 广播预测结果
    W->>C: 推送实时数据
    
    Note over A,C: 状态查询
    C->>A: GET /api/status
    A->>C: 返回服务状态
```

## 6. 部署架构图

```mermaid
graph TB
    subgraph "开发环境"
        A[Python环境<br/>FastAPI + Darts]
        B[Jupyter Notebook<br/>模型训练]
        C[Go环境<br/>数据模拟器]
    end
    
    subgraph "生产环境"
        D[FastAPI服务<br/>端口8000]
        E[Web界面<br/>静态文件服务]
        F[WebSocket<br/>实时通信]
        G[模型文件<br/>tft_model.pt]
    end
    
    subgraph "数据存储"
        H[MySQL数据库<br/>历史数据]
        I[内存缓存<br/>实时数据]
    end
    
    A --> D
    B --> G
    C --> D
    D --> E
    D --> F
    D --> G
    D --> I
    H --> I
    
    style D fill:#e8f5e8
    style G fill:#e1f5fe
    style I fill:#fff3e0
```

## 7. 错误处理机制

```mermaid
graph TD
    A[数据接收] --> B{数据验证}
    B -->|有效| C[数据缓存]
    B -->|无效| D[返回错误信息]
    
    C --> E{数据长度检查}
    E -->|≥24点| F[模型预测]
    E -->|<24点| G[返回状态信息]
    
    F --> H{预测成功}
    H -->|成功| I[WebSocket推送]
    H -->|失败| J[记录错误日志]
    
    I --> K[客户端接收]
    J --> L[错误恢复机制]
    
    style B fill:#fff3e0
    style F fill:#e8f5e8
    style I fill:#e3f2fd
```

## 8. 性能优化策略

```mermaid
graph LR
    subgraph "数据优化"
        A[内存缓存<br/>减少数据库查询]
        B[数据标准化<br/>提高模型精度]
        C[时间序列重采样<br/>统一数据频率]
    end
    
    subgraph "模型优化"
        D[批量预测<br/>提高吞吐量]
        E[模型持久化<br/>避免重复加载]
        F[异步处理<br/>非阻塞预测]
    end
    
    subgraph "网络优化"
        G[WebSocket<br/>实时推送]
        H[JSON压缩<br/>减少传输量]
        I[连接池<br/>复用连接]
    end
    
    A --> D
    B --> E
    C --> F
    D --> G
    E --> H
    F --> I
    
    style A fill:#e8f5e8
    style D fill:#e3f2fd
    style G fill:#fff3e0
```

## 技术亮点

1. **先进的时间序列模型**: TFT (Temporal Fusion Transformer) 能够处理多变量时间序列
2. **实时预测系统**: 基于WebSocket的实时数据推送
3. **多站点支持**: 支持多个南极气象站的独立预测
4. **完整的数据管道**: 从历史数据训练到实时预测的完整流程
5. **高性能架构**: 内存缓存 + 异步处理 + 批量预测
6. **易于扩展**: 模块化设计，便于添加新的气象站或特征
