import 'dart:math';
import 'package:langchain_core/chat_models.dart';
import 'package:langchain_core/language_models.dart';
import 'package:langchain_core/prompts.dart';
import 'package:langchain_llamacpp/src/types.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

class ChatLlamacpp extends BaseChatModel<ChatLlamaOptions>{
  late LlamaModelParams mParams;
  late LlamaCtxParams cParams;
  late Llama llama;
  
  ChatLlamacpp({
    required final String modelPath,
    required super.defaultOptions,
  }){
    mParams = LlamaModelParams(nGpuLayers: defaultOptions.numGpuLayers);
    cParams = LlamaCtxParams(nCtx: defaultOptions.numCtx, nThreads: 8, nThreadsbatch: 8);
    llama = Llama(mParams: mParams, cParams: cParams);

    llama.loadModel(modelPath);
  }

  @override
  String get modelType => 'llama.cpp';

  @override
  void close(){
    llama.dispose();
  }

  void stop(){
    llama.sendStop();
  }

  @override
  Future<ChatResult> invoke(
    final PromptValue input, {
    final ChatLlamaOptions? options 
  }) async{
    final params = options ?? super.defaultOptions;
    final List<Map<String, String>> chatMessages = [];
    for(final msg in input.toChatMessages()){
      chatMessages.add({'role':
        switch(msg){
          AIChatMessage _ => 'assistant',
          HumanChatMessage _ => 'user',
          SystemChatMessage _ => 'system',
          _ => throw Exception('Tipo de mensagem não suportado')
        },
        'content': msg.contentAsString
      });
    }

    final formated = llama.formatWithTemplate(chatMessages);

    //List<String> acumulated = List.generate(chatMessages.length, (i) => chatMessages[i].contentAsString);
    var result = await llama.generate(
      formated,
      isIsolated: true,
      temp: params.temperature,
      topK: params.topK,
      topP: params.topP
    );

    final promptUsage = llama.tokenize(formated).$1.length;
    
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
    final params = options ?? super.defaultOptions;
    final List<Map<String, String>> chatMessages = [];
    for(final msg in input.toChatMessages()){
      chatMessages.add({'role':
        switch(msg){
          AIChatMessage _ => 'assistant',
          HumanChatMessage _ => 'user',
          SystemChatMessage _ => 'system',
          _ => throw Exception('Tipo de mensagem não suportado')
        },
        'content': msg.contentAsString
      });
    }

    final formated = llama.formatWithTemplate(chatMessages);
    
    //final List<String> msgContent = List.generate(chatMessages.length, (i) =>chatMessages[i].contentAsString);

    final promptUsage = llama.tokenize(formated).$1.length;
    final uuid = List.generate(8, (_) => Random.secure().nextInt(16).toRadixString(16)).join();

    await for(final resp in llama.generateStreamed(
      formated,
      temp: params.temperature,
      topK: params.topK,
      topP: params.topP
    )){
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