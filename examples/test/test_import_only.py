"""Test that imports work correctly without making API calls."""

def test_imports():
    """Test that all necessary imports work."""
    from deepagents import create_deep_agent
    from deepagents.chat_models.openai_compat import CompatibleChatOpenAI
    from langchain_core.messages import HumanMessage
    
    # Test that we can create the model instance
    model = CompatibleChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        api_key="test-key",
        base_url="http://127.0.0.1:8080/chatylserver",
        temperature=0.7,
    )
    
    # Test that we can create the agent
    agent = create_deep_agent(model=model)
    
    # Test that we can create a message
    message = HumanMessage(content="Test message")
    
    print("All imports and basic object creation successful!")
    assert model is not None
    assert agent is not None
    assert message is not None

if __name__ == "__main__":
    test_imports()