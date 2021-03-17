import ast
import tulipy
import numpy as np
import octobot_trading.api as trading_api
import octobot_evaluators.util as evaluators_util
import octobot_evaluators.evaluators as evaluators
import octobot_commons.constants as commons_constants
import octobot_tentacles_manager.api as tentacles_manager_api


class GenomeEvaluator(evaluators.TAEvaluator):

    N_FEATURES: int = 7
    GENOME_SIGNAL_LEN: int = 10
    GENOME_LEN: int = N_FEATURES * GENOME_SIGNAL_LEN

    def __init__(self, tentacles_setup_config):
        super().__init__(tentacles_setup_config)
        genome: list = None
        self.genome: np.ndarray = None
        self.bb_period: int = 14
        self.rsi_period: int = 9
        self.fast_ema_period: int = 10
        self.slow_ema_period: int = 50
        self.minimum_period: int = max([
            self.bb_period,
            self.rsi_period,
            self.fast_ema_period,
            self.slow_ema_period,
        ])
        self.evaluator_config = tentacles_manager_api.get_tentacle_config(
            self.tentacles_setup_config, self.__class__)
        try:
            genome = ast.literal_eval(self.evaluator_config["genome"])
        except Exception as e:
            self.logger.error(f"Error when parsing genome: {e}")
            genome = [0 for _ in range(self.GENOME_LEN)]
        if len(genome) != self.GENOME_LEN:
            self.logger.error(f"Genome must be of size {self.GENOME_LEN}!")
            genome = [0 for _ in range(self.GENOME_LEN)]
        self.genome = np.array(genome)

    async def ohlcv_callback(self, exchange: str, exchange_id: str,
                             cryptocurrency: str, symbol: str, time_frame, candle, inc_in_construction_data):
        candle_data = trading_api.get_symbol_close_candles(self.get_exchange_symbol_data(exchange, exchange_id, symbol),
                                                           time_frame,
                                                           include_in_construction=inc_in_construction_data)
        await self.evaluate(cryptocurrency, symbol, time_frame, candle_data, candle)

    async def evaluate(self, cryptocurrency, symbol, time_frame, candle_data, candle):
        if candle_data is not None and len(candle_data) > self.minimum_period:
            lbb, mbb, ubb = tulipy.bbands(candle_data, period=self.bb_period)
            rsi = tulipy.rsi(candle_data, period=self.rsi_period)
            fast_ema = tulipy.ema(candle_data, period=self.fast_ema_period)
            slow_ema = tulipy.ema(candle_data, period=self.slow_ema_period)

            min_len = min([
                len(lbb),
                len(mbb),
                len(ubb),
                len(rsi),
                len(fast_ema),
                len(slow_ema),
                len(candle_data),
            ])

            if (min_len > self.GENOME_SIGNAL_LEN):

                is_nan = any([
                    np.isnan(lbb[-self.GENOME_SIGNAL_LEN:]).any(),
                    np.isnan(mbb[-self.GENOME_SIGNAL_LEN:]).any(),
                    np.isnan(ubb[-self.GENOME_SIGNAL_LEN:]).any(),
                    np.isnan(rsi[-self.GENOME_SIGNAL_LEN:]).any(),
                    np.isnan(fast_ema[-self.GENOME_SIGNAL_LEN:]).any(),
                    np.isnan(slow_ema[-self.GENOME_SIGNAL_LEN:]).any(),
                    np.isnan(candle_data[-self.GENOME_SIGNAL_LEN:]).any(),
                ])

                if not is_nan:
                    features: np.ndarray = np.concatenate([
                        lbb[-self.GENOME_SIGNAL_LEN:],
                        mbb[-self.GENOME_SIGNAL_LEN:],
                        ubb[-self.GENOME_SIGNAL_LEN:],
                        rsi[-self.GENOME_SIGNAL_LEN:],
                        fast_ema[-self.GENOME_SIGNAL_LEN:],
                        slow_ema[-self.GENOME_SIGNAL_LEN:],
                        candle_data[-self.GENOME_SIGNAL_LEN:]
                    ])

                    try:
                        assert features.shape[0] == self.GENOME_LEN
                    except AssertionError:
                        self.logger.error(
                            f"Features and genome length differs: (Genome: {self.GENOME_LEN}), (Features: {features.shape[0]})")
                        return

                    self.eval_note = np.sum(self.genome * features)
                    await self.evaluation_completed(cryptocurrency, symbol, time_frame,
                                                    eval_time=evaluators_util.get_eval_time(full_candle=candle,
                                                                                            time_frame=time_frame))
                    return

        self.eval_note = commons_constants.START_PENDING_EVAL_NOTE
        await self.evaluation_completed(cryptocurrency, symbol, time_frame,
                                        eval_time=evaluators_util.get_eval_time(full_candle=candle,
                                                                                time_frame=time_frame))

    @classmethod
    def get_is_symbol_wildcard(cls) -> bool:
        """
        :return: True if the evaluator is not symbol dependant else False
        """
        return False

    @classmethod
    def get_is_time_frame_wildcard(cls) -> bool:
        """
        :return: True if the evaluator is not time_frame dependant else False
        """
        return False
