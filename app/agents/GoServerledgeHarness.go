package main

// Adattatore generico per il base Executor di ServerlEdge.
// Il sorgente Go validato e' compilato senza modifiche accanto a questo file.
// Il target e' sostituito con l'entry point verificato dall'orchestratore.

import (
    "encoding/json"
    "errors"
    "fmt"
    "os"
    "reflect"
)

var functionTarget = reflect.ValueOf(__REDATASET_ENTRY_POINT__)
var errorType = reflect.TypeOf((*error)(nil)).Elem()

func runServerledgeFunction() error {
    paramsFile := os.Getenv("PARAMS_FILE")
    resultFile := os.Getenv("RESULT_FILE")
    if paramsFile == "" || resultFile == "" {
        return errors.New("PARAMS_FILE o RESULT_FILE mancante")
    }
    contents, err := os.ReadFile(paramsFile)
    if err != nil {
        return fmt.Errorf("lettura PARAMS_FILE: %w", err)
    }
    var params map[string]json.RawMessage
    if err := json.Unmarshal(contents, &params); err != nil {
        return fmt.Errorf("parametri non JSON: %w", err)
    }
    rawArgs, ok := params["args"]
    if !ok {
        return errors.New("chiave args mancante")
    }
    var argsJSON []json.RawMessage
    if err := json.Unmarshal(rawArgs, &argsJSON); err != nil {
        return fmt.Errorf("args non e' un array JSON: %w", err)
    }
    signature := functionTarget.Type()
    if signature.IsVariadic() || signature.NumIn() != len(argsJSON) || signature.NumOut() < 1 || signature.NumOut() > 2 {
        return errors.New("firma Go non compatibile con la chiamata posizionale")
    }
    if signature.NumOut() == 2 && !signature.Out(1).Implements(errorType) {
        return errors.New("la seconda risposta Go non implementa error")
    }
    args := make([]reflect.Value, len(argsJSON))
    for i, raw := range argsJSON {
        holder := reflect.New(signature.In(i))
        if err := json.Unmarshal(raw, holder.Interface()); err != nil {
            return fmt.Errorf("argomento %d: %w", i, err)
        }
        args[i] = holder.Elem()
    }
    result := functionTarget.Call(args)
    if len(result) == 2 {
        if resultError := result[1].Interface().(error); resultError != nil {
            return resultError
        }
    }
    encoded, err := json.Marshal(result[0].Interface())
    if err != nil {
        return fmt.Errorf("output non serializzabile: %w", err)
    }
    if err := os.WriteFile(resultFile, encoded, 0600); err != nil {
        return fmt.Errorf("scrittura RESULT_FILE: %w", err)
    }
    return nil
}

func main() {
    if err := runServerledgeFunction(); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
}
