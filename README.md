# Um port em C++ da demonstração FLIP do Ten Minute Physics por Matthias Müller com recursos adicionais

Demonstração/vídeo original: https://matthias-research.github.io/pages/tenMinutePhysics/

Uma portabilidade em C++ da demonstração de uma simulação de fluido FLIP (Fluido-Partícula Implícita) em JavaScript feita por Matthias Müller, usando OpenGL para renderização.

## O que é FLIP?

FLIP (FLuid Implicit Particle) é um método híbrido de simulação de fluidos que combina os melhores aspectos das abordagens baseadas em grade e em partículas:

- **Velocidade baseada em grade**: Utiliza uma grade regular (grade MAC) para resolver equações de fluidos incompressíveis com eficiência
- **Advecção baseada em partículas**: As partículas carregam velocidade e outras propriedades, evitando a difusão numérica
- **Transferência híbrida**: As velocidades são transferidas entre as partículas e a grade a cada passo de tempo usando interpolação
- **Combinação PIC/FLIP**: Combina PIC (Partícula na Célula) para estabilidade e FLIP para preservação de detalhes

Esta abordagem proporciona:
- **Incompressibilidade**: A resolução adequada da pressão garante a conservação do volume
- **Preservação de detalhes**: As partículas mantêm o movimento e a vorticidade em escala fina
- **Estabilidade**: A resolução baseada em grade evita problemas de agrupamento de partículas
- **Eficiência**: Mais rápido do que métodos de partículas puras para simulações de grande porte

O pipeline de simulação: integrar partículas → separar partículas → lidar com colisões → transferir para a grade → resolver a pressão → transferir de volta para as partículas.

## Recursos

- Simulação de fluidos FLIP/PIC com método híbrido de partículas e grade
- Manipulação interativa de obstáculos com arrastar do mouse
- Visualização de partículas e grade em tempo real com mapeamento de cores
- Parâmetros de simulação ajustáveis ​​via interface ImGui
- Controle vetorial de gravidade 2D (componentes X e Y)
- Separação de partículas e tratamento de colisões
- Compensação de deriva baseada em densidade
- Suporte multiplataforma (Linux, macOS, Windows)
- Teoricamente funciona imediatamente (não tenho certeza sobre macOS)

## Licença

- Licença MIT (mesma da versão original em JavaScript)

Eu traduzi o readme com o google tradutor, então não tá muito bom, talvez em outro momento eu reescreva do zero
