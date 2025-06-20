from yacs.config import CfgNode as CN


class ConfigDUT:
    def __init__(self):
        self.cfg = CN()
        self.cfg.name = ''
        self.cfg.stage1_ckpt = None
        self.cfg.restore_ckpt = None
        self.cfg.lr = 0.0
        self.cfg.wdecay = 0.0
        self.cfg.batch_size = 0
        self.cfg.num_steps = 0
        self.cfg.deltaA = 30.0
        self.cfg.lowRes = False
        self.cfg.withNormal = False

        self.cfg.noRGBInput = False
        self.cfg.noNormalInput = False
        self.cfg.sparseMotion = False

        self.cfg.withGan = False
        self.cfg.saveDebug = False
        self.cfg.imgScale = -1.0
        self.cfg.discPatchSize = 16
        self.cfg.withRotReg = False
        self.cfg.withRelaRot = False
        self.cfg.withMultiLayer = False
        self.cfg.withMultiScale = False
        self.cfg.withMultiLevel = False
        self.cfg.withPostScale = False
        self.cfg.addPostScale = 1.0
        self.cfg.saveCSV = False
        self.cfg.level = -1
        self.cfg.lrDecayStep = 2000000
        self.cfg.withDeformNormal = False
        self.cfg.withStepDecay = False
        self.cfg.removeNormalMap = False
        self.cfg.withOccAug = False
        self.cfg.sparse = False

        self.cfg.worldLap = False
        self.cfg.deltaType = ''


        self.cfg.weightColor = 1.0
        self.cfg.weightSSIM = 0.1
        self.cfg.weightMRF = 0.01
        self.cfg.weightChamfer = 0.0
        self.cfg.weightReg = 0.005

        self.cfg.weightLap = 1.0
        self.cfg.weightIso = 0.1
        self.cfg.weightNmlCons = 0.0

        self.cfg.ckpt1 = None
        self.cfg.ckpt2 = None
        self.cfg.rootDir = ''
        self.cfg.outDir = ''

        self.cfg.dataset = CN()
        self.cfg.dataset.source_id = None
        self.cfg.dataset.train_novel_id = None
        self.cfg.dataset.val_novel_id = None
        self.cfg.dataset.use_hr_img = None
        self.cfg.dataset.use_processed_data = None
        self.cfg.dataset.data_root = ''
        self.cfg.dataset.baseDir = ''
        # gsussian render settings
        self.cfg.dataset.bg_color = [0, 0, 0]
        self.cfg.dataset.is_white_background = False
        self.cfg.dataset.zfar = 100.0
        self.cfg.dataset.znear = 0.01
        self.cfg.dataset.trans = [0.0, 0.0, 0.0]
        self.cfg.dataset.scale = 1.0
        self.cfg.dataset.activeCameraIdxs = []
        self.cfg.dataset.testCameraIdxs = []
        self.cfg.dataset.condCameraIdxs = []
        self.cfg.dataset.subject = 'SubjectXXXX'
        self.cfg.dataset.cloth = 'tight or loose'
        self.cfg.dataset.fIdxs_test = []
        self.cfg.dataset.fIdxs_train = []
        # self.cfg.dataset.texRes = 512
        # self.cfg.dataset.texResCano = 512
        self.cfg.dataset.texResGeo = 512
        self.cfg.dataset.texResGau = 512
        self.cfg.dataset.texResCano = 512




        self.cfg.gaussian = CN()
        self.cfg.gaussian.sh_degree = 3
        self.cfg.gaussian.num_steps = 1000000
        self.cfg.gaussian.warmup_steps = 15000
        self.cfg.gaussian.lr = 1e-4
        self.cfg.gaussian.with2D = False



        self.cfg.raft = CN()
        self.cfg.raft.mixed_precision = None
        self.cfg.raft.train_iters = 0
        self.cfg.raft.val_iters = 0
        self.cfg.raft.corr_implementation = 'reg_cuda'  # or 'reg'
        self.cfg.raft.corr_levels = 4
        self.cfg.raft.corr_radius = 4
        self.cfg.raft.n_downsample = 3
        self.cfg.raft.n_gru_layers = 1
        self.cfg.raft.slow_fast_gru = None
        self.cfg.raft.encoder_dims = [64, 96, 128]
        self.cfg.raft.hidden_dims = [128]*3

        self.cfg.gsnet = CN()
        self.cfg.gsnet.encoder_dims = None
        self.cfg.gsnet.decoder_dims = None
        self.cfg.gsnet.parm_head_dim = None

        self.cfg.record = CN()
        self.cfg.record.ckpt_path = None
        self.cfg.record.show_path = None
        self.cfg.record.logs_path = None
        self.cfg.record.file_path = None
        self.cfg.record.debug_path = None
        self.cfg.record.loss_freq = 0
        self.cfg.record.loss_freq1 = 0
        self.cfg.record.loss_freq2 = 0
        self.cfg.record.eval_freq = 0

    def get_cfg(self):
        return self.cfg.clone()
    
    def load(self, config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
        self.cfg.freeze()
