import "dotenv/config";
import readline from "node:readline";
import Anthropic from "@anthropic-ai/sdk";

if (!process.env.ANTHROPIC_API_KEY) {
  console.error("Missing ANTHROPIC_API_KEY — add it to your .env file.");
  process.exit(1);
}

const MODEL = "claude-sonnet-5";
const MAX_TOKENS = 1024;

const client = new Anthropic();

const calculateTool = {
  name: "calculate",
  description:
    "Perform an arithmetic operation on two numbers. Use this for every " +
    "calculation instead of computing the result yourself.",
  input_schema: {
    type: "object",
    properties: {
      operation: {
        type: "string",
        enum: ["add", "subtract", "multiply", "divide"],
        description: "The arithmetic operation to perform.",
      },
      a: { type: "number", description: "The left-hand operand." },
      b: { type: "number", description: "The right-hand operand." },
    },
    required: ["operation", "a", "b"],
  },
};

// The result is computed here in JavaScript — Claude never supplies it.
// Returns a partial tool_result block (content + optional is_error).
function runCalculate(input) {
  const { operation, a, b } = input ?? {};

  if (typeof a !== "number" || Number.isNaN(a) || typeof b !== "number" || Number.isNaN(b)) {
    return { content: "Error: both 'a' and 'b' must be numbers.", is_error: true };
  }

  if (operation === "divide" && b === 0) {
    return { content: "Error: division by zero is undefined.", is_error: true };
  }

  switch (operation) {
    case "add":
      return { content: String(a + b) };
    case "subtract":
      return { content: String(a - b) };
    case "multiply":
      return { content: String(a * b) };
    case "divide":
      return { content: String(a / b) };
    default:
      return {
        content: `Error: unknown operation "${operation}". Expected add, subtract, multiply, or divide.`,
        is_error: true,
      };
  }
}

function printText(message) {
  const text = message.content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();

  if (text) console.log(`\nClaude: ${text}\n`);
  else console.log("\nClaude: (no text in response)\n");

  if (message.stop_reason === "max_tokens") {
    console.log(`(response truncated at max_tokens=${MAX_TOKENS})\n`);
  }
}

// Full conversation history, preserved across turns.
const messages = [];

async function handleTurn(userInput) {
  messages.push({ role: "user", content: userInput });

  while (true) {
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: MAX_TOKENS,
      tools: [calculateTool],
      tool_choice: { type: "auto" },
      messages,
    });

    // Append the whole content array so tool_use (and thinking) blocks are
    // echoed back verbatim on the next request.
    messages.push({ role: "assistant", content: response.content });

    if (response.stop_reason !== "tool_use") {
      printText(response);
      return;
    }

    const toolUses = response.content.filter((block) => block.type === "tool_use");
    const toolResults = toolUses.map((toolUse) => {
      const { content, is_error } = runCalculate(toolUse.input);
      console.log(
        `  [calculate] ${JSON.stringify(toolUse.input)} -> ${content}`,
      );
      return {
        type: "tool_result",
        tool_use_id: toolUse.id,
        content,
        ...(is_error ? { is_error: true } : {}),
      };
    });

    // All tool_results for one assistant turn go in a single user message.
    messages.push({ role: "user", content: toolResults });
  }
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: "You: ",
});

console.log(`Calculator agent ready (model: ${MODEL}). Type "exit" to quit.\n`);
rl.prompt();

// Iterating the interface pauses input while a turn is in flight, so the
// loop works the same whether stdin is a TTY or a pipe.
for await (const line of rl) {
  const input = line.trim();

  if (!input) {
    rl.prompt();
    continue;
  }
  if (input.toLowerCase() === "exit" || input.toLowerCase() === "quit") break;

  try {
    await handleTurn(input);
  } catch (error) {
    if (error instanceof Anthropic.RateLimitError) {
      console.error("\nRate limited — try again in a moment.\n");
    } else if (error instanceof Anthropic.AuthenticationError) {
      console.error("\nAuthentication failed — check ANTHROPIC_API_KEY.\n");
    } else if (error instanceof Anthropic.APIError) {
      console.error(`\nAPI error ${error.status}: ${error.message}\n`);
    } else {
      console.error(`\nUnexpected error: ${error.message}\n`);
    }
    // Drop the partial turn so history stays a valid alternating sequence.
    while (messages.length && messages.at(-1).role !== "assistant") messages.pop();
  }

  rl.prompt();
}

rl.close();
console.log("Goodbye.");
