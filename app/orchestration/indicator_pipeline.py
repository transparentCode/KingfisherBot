import logging
from typing import List, Any, Dict

from app.models.indicator_context import IndicatorExecutionContext
from app.orchestration.indicator.indicator_orchestrator import BaseIndicatorOrchestrator


class IndicatorPipeline:
    def __init__(self):
        self.orchestrators: List[BaseIndicatorOrchestrator] = []
        self.result_processors: List[Any] = []
        self.logger = logging.getLogger("app")

    def add_orchestrator(self, orchestrator: BaseIndicatorOrchestrator):
        self.orchestrators.append(orchestrator)

    def add_result_processor(self, processor):
        self.result_processors.append(processor)

    async def execute_pipeline(self, context: IndicatorExecutionContext) -> Dict[str, Any]:
        self.logger.info(f"Pipeline execution started for {context.asset}")
        all_results = {}

        if not self.orchestrators:
            self.logger.warning("No orchestrators registered in pipeline!")

        for orchestrator in self.orchestrators:
            try:
                orchestrator_name = orchestrator.__class__.__name__
                self.logger.info(f"Executing {orchestrator_name} for {context.asset}")

                result = await orchestrator.execute(context)
                if result:
                    all_results.update(result)
                    self.logger.info(f"Completed {orchestrator_name}: {len(result)} indicators calculated")
                else:
                    self.logger.warning(f"{orchestrator_name} returned no results")

            except Exception as e:
                self.logger.error(f"Error in orchestrator {orchestrator.__class__.__name__}: {e}", exc_info=True)
                continue

        self.logger.info(f"Total results collected: {len(all_results)}")

        # Inject MTF data for processors that need raw context (e.g. BotBrain)
        all_results['__mtf_data__'] = context.data_cache

        # Process results through processors
        if not self.result_processors:
            self.logger.warning("No result processors registered in pipeline!")
        
        for processor in self.result_processors:
            try:
                processor_name = processor.__class__.__name__
                self.logger.info(f"Calling processor: {processor_name}")
                await processor.process_results(context.asset, all_results)
                self.logger.info(f"Processor {processor_name} completed")
            except Exception as e:
                self.logger.error(f"Error in result processor {processor.__class__.__name__}: {e}", exc_info=True)

        return all_results