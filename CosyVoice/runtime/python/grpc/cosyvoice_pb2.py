# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: cosyvoice.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x63osyvoice.proto\x12\tcosyvoice\"\xfb\x01\n\x07Request\x12,\n\x0bsft_request\x18\x01 \x01(\x0b\x32\x15.cosyvoice.sftRequestH\x00\x12\x37\n\x11zero_shot_request\x18\x02 \x01(\x0b\x32\x1a.cosyvoice.zeroshotRequestH\x00\x12?\n\x15\x63ross_lingual_request\x18\x03 \x01(\x0b\x32\x1e.cosyvoice.crosslingualRequestH\x00\x12\x36\n\x10instruct_request\x18\x04 \x01(\x0b\x32\x1a.cosyvoice.instructRequestH\x00\x42\x10\n\x0eRequestPayload\".\n\nsftRequest\x12\x0e\n\x06spk_id\x18\x01 \x01(\t\x12\x10\n\x08tts_text\x18\x02 \x01(\t\"N\n\x0fzeroshotRequest\x12\x10\n\x08tts_text\x18\x01 \x01(\t\x12\x13\n\x0bprompt_text\x18\x02 \x01(\t\x12\x14\n\x0cprompt_audio\x18\x03 \x01(\x0c\"=\n\x13\x63rosslingualRequest\x12\x10\n\x08tts_text\x18\x01 \x01(\t\x12\x14\n\x0cprompt_audio\x18\x02 \x01(\x0c\"J\n\x0finstructRequest\x12\x10\n\x08tts_text\x18\x01 \x01(\t\x12\x0e\n\x06spk_id\x18\x02 \x01(\t\x12\x15\n\rinstruct_text\x18\x03 \x01(\t\"\x1d\n\x08Response\x12\x11\n\ttts_audio\x18\x01 \x01(\x0c\x32\x45\n\tCosyVoice\x12\x38\n\tInference\x12\x12.cosyvoice.Request\x1a\x13.cosyvoice.Response\"\x00\x30\x01\x42\tZ\x07protos/b\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'cosyvoice_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z\007protos/'
  _globals['_REQUEST']._serialized_start=31
  _globals['_REQUEST']._serialized_end=282
  _globals['_SFTREQUEST']._serialized_start=284
  _globals['_SFTREQUEST']._serialized_end=330
  _globals['_ZEROSHOTREQUEST']._serialized_start=332
  _globals['_ZEROSHOTREQUEST']._serialized_end=410
  _globals['_CROSSLINGUALREQUEST']._serialized_start=412
  _globals['_CROSSLINGUALREQUEST']._serialized_end=473
  _globals['_INSTRUCTREQUEST']._serialized_start=475
  _globals['_INSTRUCTREQUEST']._serialized_end=549
  _globals['_RESPONSE']._serialized_start=551
  _globals['_RESPONSE']._serialized_end=580
  _globals['_COSYVOICE']._serialized_start=582
  _globals['_COSYVOICE']._serialized_end=651
# @@protoc_insertion_point(module_scope)
