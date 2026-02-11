#!/bin/bash
# Quick Start Script for Testing
# This script helps you get started with testing

echo "=================================="
echo "InsightFace Embedder - Quick Start"
echo "=================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing dependencies..."
    pip install -r requirements.txt
else
    echo "✅ pytest is installed"
fi

echo ""
echo "Choose what to run:"
echo "1. Run all tests (including slow integration tests)"
echo "2. Run only fast unit tests (RECOMMENDED for first run)"
echo "3. Run with coverage report"
echo "4. Run specific test class"
echo "5. Show all available tests"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Running all tests..."
        pytest -v
        ;;
    2)
        echo ""
        echo "Running fast unit tests only..."
        pytest -v -m "not slow"
        ;;
    3)
        echo ""
        echo "Running tests with coverage..."
        pytest --cov=src --cov-report=html --cov-report=term-missing
        echo ""
        echo "✅ Coverage report generated at: htmlcov/index.html"
        ;;
    4)
        echo ""
        echo "Available test classes:"
        echo "  - TestEmbedderUnitTests"
        echo "  - TestEmbedderInitialization"
        echo "  - TestFaceEmbeddingSchema"
        echo "  - TestCustomExceptions"
        echo "  - TestEdgeCases"
        echo "  - TestParametrized"
        echo ""
        read -p "Enter class name: " classname
        pytest -v tests/test_embedder.py::$classname
        ;;
    5)
        echo ""
        echo "Collecting all tests..."
        pytest --collect-only
        ;;
    *)
        echo "Invalid choice. Running fast tests by default..."
        pytest -v -m "not slow"
        ;;
esac

echo ""
echo "=================================="
echo "Testing complete!"
echo "=================================="
echo ""
echo "Quick commands:"
echo "  pytest                    # Run all tests"
echo "  pytest -m 'not slow'      # Skip slow tests"
echo "  pytest -v                 # Verbose output"
echo "  pytest -x                 # Stop on first failure"
echo "  pytest -k 'test_name'     # Run tests matching name"
echo ""
echo "For more info, see README.md"