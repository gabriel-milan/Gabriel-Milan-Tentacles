import ast
import tulipy
import numpy as np
import octobot_trading.api as trading_api
import octobot_evaluators.util as evaluators_util
import octobot_evaluators.evaluators as evaluators
import octobot_commons.constants as commons_constants
import octobot_tentacles_manager.api as tentacles_manager_api


class GenomeEvaluator(evaluators.TAEvaluator):

    N_FEATURES: int = 6
    GENOME_SIGNAL_LEN: int = 10
    GENOME_LEN: int = N_FEATURES * GENOME_SIGNAL_LEN
    THRESHOLD: float = 0.5

    def __init__(self, tentacles_setup_config):
        super().__init__(tentacles_setup_config)
        genome: list = None
        self._bought: bool = False
        self.genome: np.ndarray = None
        self.bb_period: int = 14
        self.rsi_period: int = 9
        self.fast_ema_period: int = 10
        self.slow_ema_period: int = 50
        self.minimum_period: int = max([
            self.bb_period,
            self.rsi_period,
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

    def buy(self):
        if not self._bought:
            self._bought = True
            return -1
        return 0

    def sell(self):
        if self._bought:
            self._bought = False
            return 1
        return 0

    def do_nothing(self):
        return 0

    async def ohlcv_callback(self, exchange: str, exchange_id: str,
                             cryptocurrency: str, symbol: str, time_frame, candle, inc_in_construction_data):
        candle_data = trading_api.get_symbol_close_candles(self.get_exchange_symbol_data(exchange, exchange_id, symbol),
                                                           time_frame,
                                                           include_in_construction=inc_in_construction_data)
        await self.evaluate(cryptocurrency, symbol, time_frame, candle_data, candle)

    async def evaluate(self, cryptocurrency, symbol, time_frame, candle_data, candle):
        if len(candle_data >= 50):
            self.logger.debug(f"candle data is {candle_data[-50:]}")
        else:
            self.logger.debug(f"candle data is {candle_data}")
        if candle_data is not None and len(candle_data) >= self.minimum_period:
            self.logger.debug(f"candle data len is {len(candle_data)}")
            lbb, mbb, ubb = tulipy.bbands(
                candle_data, period=self.bb_period, stddev=2)
            lbb_len = len(lbb)
            mbb_len = len(mbb)
            ubb_len = len(ubb)
            lbb = (lbb - candle_data[-lbb_len:]) / candle_data[-lbb_len:]
            mbb = (mbb - candle_data[-mbb_len:]) / candle_data[-mbb_len:]
            ubb = (ubb - candle_data[-ubb_len:]) / candle_data[-ubb_len:]
            rsi = tulipy.rsi(candle_data, period=self.rsi_period) / 100
            fast_ema = tulipy.ema(candle_data, period=self.fast_ema_period)
            fe_len = len(fast_ema)
            fast_ema = (
                fast_ema - candle_data[-fe_len:]) / candle_data[-fe_len:]
            slow_ema = tulipy.ema(candle_data, period=self.slow_ema_period)
            se_len = len(slow_ema)
            slow_ema = (
                slow_ema - candle_data[-se_len:]) / candle_data[-se_len:]

            min_len = min([
                len(lbb),
                len(mbb),
                len(ubb),
                len(rsi),
                len(fast_ema),
                len(slow_ema),
            ])

            if (min_len >= self.GENOME_SIGNAL_LEN):

                lbb = lbb[-self.GENOME_SIGNAL_LEN:]
                mbb = mbb[-self.GENOME_SIGNAL_LEN:]
                ubb = ubb[-self.GENOME_SIGNAL_LEN:]
                rsi = rsi[-self.GENOME_SIGNAL_LEN:]
                fast_ema = fast_ema[-self.GENOME_SIGNAL_LEN:]
                slow_ema = slow_ema[-self.GENOME_SIGNAL_LEN:]

                self.logger.debug(f"lbb is {lbb}")
                self.logger.debug(f"mbb is {mbb}")
                self.logger.debug(f"ubb is {ubb}")
                self.logger.debug(f"rsi is {rsi}")
                self.logger.debug(f"fast_ema is {fast_ema}")
                self.logger.debug(f"slow_ema is {slow_ema}")

                is_nan = any([
                    np.isnan(lbb).any(),
                    np.isnan(mbb).any(),
                    np.isnan(ubb).any(),
                    np.isnan(rsi).any(),
                    np.isnan(fast_ema).any(),
                    np.isnan(slow_ema).any(),
                ])

                if not is_nan:
                    features: np.ndarray = np.concatenate([
                        lbb,
                        mbb,
                        ubb,
                        rsi,
                        fast_ema,
                        slow_ema,
                    ])

                    try:
                        assert features.shape[0] == self.GENOME_LEN
                    except AssertionError:
                        raise Exception(
                            f"Features and genome length differs: (Genome: {self.GENOME_LEN}), (Features: {features.shape[0]})")

                    result = np.sum(self.genome * features)
                    self.logger.debug(f"sum is {result}")

                    if (result >= self.THRESHOLD):
                        self.logger.debug(
                            "sum is greater than +threshold, let's buy!")
                        self.eval_note = self.buy()
                    elif (result <= -self.THRESHOLD):
                        self.logger.debug(
                            "sum is lower than -threshold, let's sell!")
                        self.eval_note = self.sell()
                    else:
                        self.logger.debug(
                            "sum is within (-threshold, +threshold), I'll do nothing")
                        self.eval_note = self.do_nothing()

                    await self.evaluation_completed(cryptocurrency, symbol, time_frame,
                                                    eval_time=evaluators_util.get_eval_time(full_candle=candle,
                                                                                            time_frame=time_frame))
                    return
                else:
                    self.logger.debug("is_nan is True, this is bad!")
            else:
                self.logger.debug(f"min_len not enough! len = {min_len}")
        else:
            self.logger.debug(
                f"candle data is not enough! len = {len(candle_data)}")
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
