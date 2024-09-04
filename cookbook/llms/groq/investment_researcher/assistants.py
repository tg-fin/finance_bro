from textwrap import dedent

from phi.assistant import Assistant
from phi.llm.groq import Groq
from phi.utils.log import logger

def get_invstment_research_assistant(
    model: str = "llama-3.1-70b-versatile",
    api_key: str="",
    debug_mode: bool = True,
) -> Assistant:
    if not api_key:
        logger.error("API key is empty")
        raise ValueError("Groq API Key is not provided. Please ensure you've entered your API key in the Streamlit app.")
    
    try:
        logger.info(f"Attempting to create Groq LLM with model: {model}")
        groq_llm = Groq(model=model, api_key=api_key)
        logger.info("Groq LLM created successfully")
        
        assistant = Assistant(
            name="investment_research_assistant_groq",
            llm=Groq(model=model, api_key=api_key),
            description="You are a Senior Investment Analyst for Goldman Sachs tasked with producing a research report for a very important client.",
            instructions=[
                "You will be provided with a stock and information from junior researchers.",
                "Carefully read the research and generate a final - Goldman Sachs worthy investment report.",
                "Make your report engaging, informative, and well-structured.",
                "When you share numbers, make sure to include the units (e.g., millions/billions) and currency.",
                "REMEMBER: This report is for a very important client, so the quality of the report is important.",
                "Make sure your report is properly formatted and follows the <report_format> provided below.",
            ],
            markdown=True,
            add_datetime_to_instructions=True,
            add_to_system_prompt=dedent(
                """
            <report_format>
            ## [Company Name]: Investment Report

            ### **Overview**
            {give a brief introduction of the company and why the user should read this report}
            {make this section engaging and create a hook for the reader}

            ### Core Metrics
            {provide a summary of core metrics and show the latest data}
            - Current price: {current price}
            - 52-week high: {52-week high}
            - 52-week low: {52-week low}
            - Market Cap: {Market Cap} in billions
            - P/E Ratio: {P/E Ratio}
            - Earnings per Share: {EPS}
            - 50-day average: {50-day average}
            - 200-day average: {200-day average}
            - Analyst Recommendations: {buy, hold, sell} (number of analysts)

            ### Financial Performance
            {provide a detailed analysis of the company's financial performance}

            ### Growth Prospects
            {analyze the company's growth prospects and future potential}

            ### News and Updates
            {summarize relevant news that can impact the stock price}

            ### Upgrades and Downgrades
            {share 2 upgrades or downgrades including the firm, and what they upgraded/downgraded to}
            {this should be a paragraph not a table}

            ### [Summary]
            {give a summary of the report and what are the key takeaways}

            ### [Recommendation]
            {provide a recommendation on the stock along with a thorough reasoning}

            DISCLAIMER: The information provided by this app is for informational purposes only and does not constitute financial or investment advice. Always conduct your own research or consult a qualified financial advisor before making investment decisions. We are not responsible for any actions taken based on the output of this app.
            
            Report generated on: {Month Date, Year (hh:mm AM/PM)}
            </report_format>
            """
            ),
            debug_mode=debug_mode,
        )
        logger.info("Assistant created successfully")
        return assistant
    except Exception as e:
        logger.error(f"Error creating assistant: {str(e)}")
        raise ValueError(f"Failed to create assistant: {str(e)}")
