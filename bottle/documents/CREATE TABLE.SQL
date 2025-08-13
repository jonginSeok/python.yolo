-- Table: class_metric
DROP TABLE IF EXISTS training_classmetric;
-- Table: training_metric
DROP TABLE IF EXISTS training_trainingmetric;
-- Table: training_session
DROP TABLE IF EXISTS training_trainingsession;


-- Table: training_session
DROP TABLE IF EXISTS training_trainingsession;
CREATE TABLE training_trainingsession (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'training',
    dataset_name VARCHAR(100) NOT NULL,
    gpu_info VARCHAR(100) NOT NULL,
    memory_info VARCHAR(50) NOT NULL,
    total_epochs INTEGER NOT NULL DEFAULT 100,
    current_epoch INTEGER NOT NULL DEFAULT 10,
    batch_size integer CHECK (batch_size >= 1 AND batch_size <= 64),		--# 배치 크기
    learning_rate FLOAT NOT NULL DEFAULT 0.01  CHECK (learning_rate >= 0.0001 AND learning_rate <= 1.0),   --# 학습률
    image_size integer NOT NULL ,  --# 이미지 크기
    optimizer VARCHAR(10) DEFAULT 'SGD',  --# 옵티마이저
    augmentation BOOLEAN  DEFAULT True ,   --# 데이터 증강 사용
		early_stopping BOOLEAN  DEFAULT True ,  --# 조기 종료 사용
		patience integer DEFAULT 10 CHECK (patience >= 1 AND patience <= 100) ,   -- # 조기 종료 patience
		description TEXT,   -- # 설명
		dataset_path VARCHAR(6000), -- 데이터셋 경로 
		config TEXT,  -- 설정정보 
		start_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_id  VARCHAR(500) NOT NULL
    --updated_at TIMESTAMP WITH TIME ZONE NULL DEFAULT CURRENT_TIMESTAMP,
    --updated_id  VARCHAR(500) NULL
)
TABLESPACE pg_default;

-- Table: training_metric
DROP TABLE IF EXISTS training_trainingmetric;
CREATE TABLE training_trainingmetric (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES training_trainingsession(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    train_loss FLOAT NOT NULL,
    val_loss FLOAT NOT NULL,
    map50 FLOAT NOT NULL,
    map95 FLOAT NOT NULL,
    precision FLOAT NOT NULL DEFAULT 0.0,
    recall FLOAT NOT NULL DEFAULT 0.0,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_id  VARCHAR(500) NOT NULL,
    --updated_at TIMESTAMP WITH TIME ZONE NULL DEFAULT CURRENT_TIMESTAMP,
    --updated_id  VARCHAR(500) NULL,
    
    UNIQUE (session_id, epoch)
)
TABLESPACE pg_default;

-- Table: class_metric
DROP TABLE IF EXISTS training_classmetric;
CREATE TABLE training_classmetric (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES training_trainingsession(id) ON DELETE CASCADE,
    index INTEGER NOT NULL,
    class_name VARCHAR(50) NOT NULL,
    precision FLOAT NULL,
    recall FLOAT NULL,
    f1_score FLOAT NULL,
    instances INTEGER NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_id  VARCHAR(500) NOT NULL,
    --updated_at TIMESTAMP WITH TIME ZONE NULL DEFAULT CURRENT_TIMESTAMP,
    --updated_id  VARCHAR(500) NULL,
    
    UNIQUE (session_id, class_name)
)
TABLESPACE pg_default;


-- Add comments to the table and its columns
COMMENT ON TABLE public.training_trainingsession IS '정보';
COMMENT ON COLUMN public.training_trainingsession.id IS '학습id';
COMMENT ON COLUMN public.training_trainingsession.model_name IS '모델명';
COMMENT ON COLUMN public.training_trainingsession.version IS '버전';
COMMENT ON COLUMN public.training_trainingsession.status IS '상태';
COMMENT ON COLUMN public.training_trainingsession.dataset_name IS '데이터셋';
COMMENT ON COLUMN public.training_trainingsession.gpu_info IS 'GPU 정보';
COMMENT ON COLUMN public.training_trainingsession.memory_info IS '메모리 정보';
COMMENT ON COLUMN public.training_trainingsession.total_epochs IS '총 에포크';
COMMENT ON COLUMN public.training_trainingsession.current_epoch IS '현재 에포크';
COMMENT ON COLUMN public.training_trainingsession.batch_size IS '배치 크기';
COMMENT ON COLUMN public.training_trainingsession.learning_rate IS '학습률';
COMMENT ON COLUMN public.training_trainingsession.image_size IS '이미지 크기';
COMMENT ON COLUMN public.training_trainingsession.optimizer IS '옵티마이저';
COMMENT ON COLUMN public.training_trainingsession.augmentation IS '데이터 증강 사용';
COMMENT ON COLUMN public.training_trainingsession.early_stopping IS '조기 종료 사용';
COMMENT ON COLUMN public.training_trainingsession.patience IS '조기 종료 patience';
COMMENT ON COLUMN public.training_trainingsession.description IS '설명';
COMMENT ON COLUMN public.training_trainingsession.dataset_path IS '데이터셋 경로'; 
COMMENT ON COLUMN public.training_trainingsession.config IS '설정정보 ';
COMMENT ON COLUMN public.training_trainingsession.start_time IS '시작 시간';
COMMENT ON COLUMN public.training_trainingsession.end_time IS '종료 시간';
COMMENT ON COLUMN public.training_trainingsession.created_at IS '생성시간';
COMMENT ON COLUMN public.training_trainingsession.created_id IS '생성자 id';
--COMMENT ON COLUMN public.training_trainingsession.updated_at IS '수정시간';
--COMMENT ON COLUMN public.training_trainingsession.updated_id IS '수정자 id';

-- Add comments to the table and its columns
COMMENT ON TABLE public.training_trainingmetric IS '측정';
COMMENT ON COLUMN public.training_trainingmetric.id IS '분류id';
COMMENT ON COLUMN public.training_trainingmetric.session_id IS '학습id';
COMMENT ON COLUMN public.training_trainingmetric.epoch IS '에포크';
COMMENT ON COLUMN public.training_trainingmetric.train_loss IS '훈련 손실';
COMMENT ON COLUMN public.training_trainingmetric.val_loss IS '검증 손실';
COMMENT ON COLUMN public.training_trainingmetric.map50 IS 'mAP@0.5';
COMMENT ON COLUMN public.training_trainingmetric.map95 IS 'mAP@0.5:0.95';
COMMENT ON COLUMN public.training_trainingmetric.precision IS '정밀도';
COMMENT ON COLUMN public.training_trainingmetric.recall IS '재현율';
COMMENT ON COLUMN public.training_trainingmetric.timestamp IS '현재 에포크';
COMMENT ON COLUMN public.training_trainingmetric.created_at IS '생성시간';
COMMENT ON COLUMN public.training_trainingmetric.created_id IS '생성자 id';
--COMMENT ON COLUMN public.training_trainingmetric.updated_at IS '수정시간';
--COMMENT ON COLUMN public.training_trainingmetric.updated_id IS '수정자 id';

-- Add comments to the table and its columns
COMMENT ON TABLE public.training_classmetric IS '분류';
COMMENT ON COLUMN public.training_classmetric.id IS '분류id';
COMMENT ON COLUMN public.training_classmetric.session_id IS '학습id';
COMMENT ON COLUMN public.training_classmetric.index IS '인덱스';
COMMENT ON COLUMN public.training_classmetric.class_name IS '분류명';
COMMENT ON COLUMN public.training_classmetric.precision IS '정밀도';
COMMENT ON COLUMN public.training_classmetric.recall IS '재현율';
COMMENT ON COLUMN public.training_classmetric.f1_score IS 'F1';
COMMENT ON COLUMN public.training_classmetric.instances IS '검증손실';
COMMENT ON COLUMN public.training_classmetric.created_at IS '생성시간';
COMMENT ON COLUMN public.training_classmetric.created_id IS '생성자 id';
--COMMENT ON COLUMN public.training_classmetric.updated_at IS '수정시간';
--COMMENT ON COLUMN public.training_classmetric.updated_id IS '수정자 id';
