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
        all_results = {}

        for orchestrator in self.orchestrators:
            try:
                orchestrator_name = orchestrator.__class__.__name__
                self.logger.debug(f"Executing {orchestrator_name} for {context.asset}")

                result = await orchestrator.execute(context)
                all_results.update(result)

                self.logger.debug(f"Completed {orchestrator_name}: {len(result)} indicators calculated")

            except Exception as e:
                self.logger.error(f"Error in orchestrator {orchestrator.__class__.__name__}: {e}")
                continue

        # Process results through processors
        for processor in self.result_processors:
            try:
                await processor.process_results(context.asset, all_results)
            except Exception as e:
                self.logger.error(f"Error in result processor {processor.__class__.__name__}: {e}")

        return all_results