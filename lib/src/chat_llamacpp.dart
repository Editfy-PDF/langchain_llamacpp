import 'dart:math';
import 'package:langchain_core/chat_models.dart';
import 'package:langchain_core/language_models.dart';
import 'package:langchain_core/prompts.dart';
import 'package:langchain_llamacpp/src/types.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

class ChatLlamacpp extends BaseChatModel<ChatLlamaOptions>{
  ChatLlamacpp({
    required final String modelPath,
    required super.defaultOptions,
  }){
    llama.loadModel(modelPath);
  }

  final Llama llama = Llama();

  @override
  String get modelType => 'llama.cpp';

  @override
  Future<ChatResult> invoke(
    final PromptValue input, {
    final ChatLlamaOptions? options 
  }) async{
    final List<ChatMessage> chatMessages = input.toChatMessages();

    List<String> acumulated = List.generate(chatMessages.length, (i) => chatMessages[i].contentAsString);
    var result = await llama.generate(acumulated.join('\n'));

    final promptUsage = llama.tokenize(acumulated.join('\n')).$1.length;
    
    return ChatResult(
      id: List.generate(8, (_) => Random.secure().nextInt(16).toRadixString(16)).join(),
      output: AIChatMessage(content: result),
      finishReason: FinishReason.unspecified,
      metadata: {
        'model': options != null ? options.model : 'unknow'
      },
      usage: LanguageModelUsage(
        promptTokens: promptUsage,
        responseTokens: llama.tokenize(result).$1.length
      ),
      streaming: false
    );
  }

  @override
  Stream<ChatResult> stream(
    final PromptValue input, {
    final ChatLlamaOptions? options
  }) async*{
    final chatMessages = input.toChatMessages();
    final List<String> msgContent = List.generate(chatMessages.length, (i) =>chatMessages[i].contentAsString);

    final promptUsage = llama.tokenize(msgContent.join('\n')).$1.length;
    final uuid = List.generate(8, (_) => Random.secure().nextInt(16).toRadixString(16)).join();

    await for(final resp in llama.generateStreamed(msgContent.join('\n'))){
      yield ChatResult(
        id: uuid,
        output: AIChatMessage(content: resp),
        finishReason: FinishReason.unspecified,
        metadata: {
          'model': options != null ? options.model : 'unknow'
        },
        usage: LanguageModelUsage(
          promptTokens: promptUsage,
          responseTokens: llama.tokenize(resp).$1.length
        ),
        streaming: true
      );
    }
  }

  @override
  Future<List<int>> tokenize(
    final PromptValue prompt, {
    final ChatLlamaOptions? options
  }) async{
    final chatMessages = prompt.toChatMessages();

    return llama.tokenize(
      List.generate(chatMessages.length, (i) => chatMessages[i].contentAsString).join('\n')
    ).$1;
  }
}